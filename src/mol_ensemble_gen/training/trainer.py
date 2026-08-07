"""DDP finetuning loop for the ESMFold2 diffusion module + temperature embedder.

Loads ESMFold2 once, keeps only ``structure_head`` (the trunk/PLM are not needed
once conditioning is cached), freezes everything except ``diffusion_module`` and a
new :class:`~.conditioning.TemperatureEmbedder`, and trains the EDM loss on
streamed mdCATH frames.

Design mirrors ``execution.py``: sharded work (here by DDP rank + DataLoader
worker), atomic checkpoints (tmp + rename), idempotent resume. Launch under
``torchrun`` for multi-GPU; runs single-process if the torchrun env is absent.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from pathlib import Path


def _ddp_env() -> tuple[int, int, int, bool]:
    """Return ``(rank, world_size, local_rank, is_distributed)`` from torchrun env."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", 0)),
            True,
        )
    return 0, 1, 0, False


def _amp_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _lr_lambda(cfg):
    """Linear warmup then cosine decay to ``lr_min_ratio`` over ``max_steps``."""
    warmup, total, floor = cfg.warmup_steps, cfg.max_steps, cfg.lr_min_ratio

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return fn


def _make_trainable():
    from torch import nn

    class _Trainable(nn.Module):
        """Wraps the only grad-bearing modules so DDP can sync their gradients.

        ``head`` is kept as a plain attribute (not a submodule) so its frozen
        params are not registered/synced; only its stateless helpers are used.
        """

        def __init__(self, diffusion_module, temp_embedder, head, scheme="edm", flow=None):
            super().__init__()
            self.diffusion_module = diffusion_module
            self.temp_embedder = temp_embedder
            object.__setattr__(self, "_head", head)
            self._scheme = scheme
            object.__setattr__(self, "_flow", flow)

        def forward(self, conditioning, gt_coords, atom_mask, temperature, generator=None):
            from .loss import diffusion_loss

            return diffusion_loss(
                self._scheme,
                self.diffusion_module, self._head, self.temp_embedder,
                conditioning, gt_coords, atom_mask, temperature,
                flow=self._flow, generator=generator,
            )

    return _Trainable


class _CondCache:
    """LRU cache of per-domain conditioning tensors on the training device.

    DataLoader workers interleave domains, so a size-1 cache would thrash; a few
    slots keep each worker's current domain resident.
    """

    def __init__(self, cache_dir, device, maxsize: int = 4):
        self.cache_dir = cache_dir
        self.device = device
        self.maxsize = maxsize
        self._store: OrderedDict[str, dict] = OrderedDict()

    def get(self, domain: str) -> dict:
        import torch

        from .featurize import load_conditioning

        if domain in self._store:
            self._store.move_to_end(domain)
            return self._store[domain]
        raw = load_conditioning(self.cache_dir, domain, device="cpu")
        cond: dict = {}
        for k, v in raw.items():
            if v is None:
                cond[k] = None
            elif v.is_floating_point():
                cond[k] = v.to(self.device, dtype=torch.float32)
            else:
                cond[k] = v.to(self.device)
        self._store[domain] = cond
        self._store.move_to_end(domain)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return cond


def _load_head(model_name: str, device: str):
    """Load ESMFold2 and return only ``structure_head`` on ``device``.

    The trunk and PLM are dropped to free GPU memory — training needs just the
    denoiser and the head's stateless alignment helpers.
    """
    import gc

    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(model_name)
    head = model.structure_head
    model.structure_head = None  # detach so deleting the model frees the rest
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return head.to(device)


def _build_denoiser(cfg, device: str):
    """Return ``(diffusion_module, geometry_helpers)`` for the configured backend.

    ``backend: "ours"`` builds :class:`~mol_ensemble_gen.model.denoiser.DiffusionModule`
    and loads the pretrained denoiser weights out of the reference model *once*,
    then throws the rest away. This is the only backend that supports native
    flow-time conditioning, and it never keeps the trunk or PLM resident.
    """
    if cfg.model.backend == "reference":
        head = _load_head(cfg.model.model_name, device)
        head.requires_grad_(False)
        return head.diffusion_module, head

    from ..model.denoiser import DenoiserConfig, DiffusionModule, GeometryOps

    dcfg = DenoiserConfig(t_conditioning=cfg.flow.t_conditioning)
    module = DiffusionModule(dcfg)

    # Pull the pretrained denoiser tensors from the reference model, then release it.
    ref_head = _load_head(cfg.model.model_name, "cpu")
    state = ref_head.diffusion_module.state_dict()
    result = module.load_state_dict(state, strict=False)
    allowed = set(module.flow_time_parameter_names()) | {
        "conditioning.t_fourier.w",
        "conditioning.t_fourier.b",
    }
    unexpected_missing = [k for k in result.missing_keys if k not in allowed]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"denoiser weight mismatch — missing {unexpected_missing[:6]}, "
            f"unexpected {list(result.unexpected_keys)[:6]}"
        )

    import gc

    import torch

    del ref_head, state
    gc.collect()
    torch.cuda.empty_cache()
    return module.to(device), GeometryOps()


def _save_checkpoint(path: Path, trainable, optim, sched, scaler, step: int, cfg_dict: dict) -> None:
    import torch

    state = {
        "global_step": step,
        "diffusion_module": trainable.diffusion_module.state_dict(),
        "temp_embedder": trainable.temp_embedder.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": cfg_dict,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _cycle(loader):
    """Infinite iterator over a (finite) DataLoader for step-based training."""
    while True:
        for item in loader:
            yield item


def _run_validation(module, cond_cache, val_iter, n_batches, amp_dtype, device, seed):
    """Average the loss over ``n_batches`` held-out micro-batches.

    Two details make the number comparable across evaluations:

    * a **fixed generator seed**, so every call draws the same noise levels and the
      curve reflects the model changing rather than the σ draw;
    * ``module.eval()`` around the pass, restored afterwards.

    Every DDP rank runs the identical batches (the val stream is built with
    ``world_size=1``) and computes the same value, so no collective is needed and
    the ranks cannot drift apart. Returns ``{}`` when there is nothing to validate.
    """
    import torch

    if val_iter is None or n_batches <= 0:
        return {}

    was_training = module.training
    module.eval()
    losses, mses = [], []
    try:
        with torch.no_grad():
            for i in range(n_batches):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break
                cond = cond_cache.get(batch.domain)
                gt = torch.from_numpy(batch.gt_coords).to(device)
                mask = torch.from_numpy(batch.atom_mask).to(device)
                # Same seed each call ⇒ same σ per position in the sequence.
                gen = torch.Generator(device=device).manual_seed(seed + i)
                with torch.autocast(
                    "cuda", dtype=amp_dtype, enabled=amp_dtype is not torch.float32
                ):
                    loss, metrics = module(cond, gt, mask, batch.temperature, generator=gen)
                losses.append(float(loss))
                mses.append(metrics["mse"])
    finally:
        if was_training:
            module.train()

    if not losses:
        return {}
    return {"loss": sum(losses) / len(losses), "mse": sum(mses) / len(mses)}


def _init_wandb(cfg, cfg_dict: dict, resume_step: int):
    """Init a W&B run on rank 0 (lazy import); return the run or ``None``.

    ``wandb`` is only imported/required when ``cfg.wandb.enabled``. The run id is
    derived from ``out_dir`` when unset so resuming a checkpoint continues the
    same run (``resume="allow"``).
    """
    wcfg = cfg.wandb
    if not wcfg.enabled:
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "wandb.enabled is true but wandb is not installed "
            "(pip install -e '.[training]' or pip install wandb)"
        ) from exc

    import re

    run_id = wcfg.id or re.sub(r"[^0-9A-Za-z_.-]+", "-", str(cfg.out_dir)).strip("-")
    run = wandb.init(
        project=wcfg.project,
        entity=wcfg.entity,
        name=wcfg.run_name,
        id=run_id,
        tags=list(wcfg.tags),
        mode=wcfg.mode,
        config=cfg_dict,
        resume="allow",
        dir=str(cfg.out_dir),
    )
    if resume_step:
        print(f"[train] wandb resuming run {run_id} at step {resume_step}", flush=True)
    return run


def train(cfg) -> None:
    """Run finetuning to completion (or to ``cfg.optim.max_steps``)."""
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    from .config import config_to_dict
    from .conditioning import build_temperature_embedder
    from .mdcath import make_dataset

    rank, world_size, local_rank, distributed = _ddp_env()
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
    torch.manual_seed(cfg.seed + rank)

    out_dir = Path(cfg.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    diffusion_module, head = _build_denoiser(cfg, device)
    diffusion_module.requires_grad_(True)
    temp_embedder = build_temperature_embedder(cfg.temperature).to(device)
    if rank == 0:
        print(
            f"[train] backend={cfg.model.backend} "
            f"t_conditioning={cfg.flow.t_conditioning if cfg.optim.scheme == 'flow' else 'n/a'}",
            flush=True,
        )

    Trainable = _make_trainable()
    trainable = Trainable(
        diffusion_module, temp_embedder, head,
        scheme=cfg.optim.scheme, flow=cfg.flow,
    ).to(device)
    if rank == 0:
        print(f"[train] scheme={cfg.optim.scheme}", flush=True)

    params = [p for p in trainable.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        params, lr=cfg.optim.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda(cfg.optim))
    amp_dtype = _amp_dtype(cfg.amp_dtype)
    scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None

    global_step = 0
    if cfg.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        # A checkpoint written before native flow-time conditioning existed has no
        # t-head tensors. Allow exactly those to be absent so an older run can be
        # continued into a t-conditioned model; anything else is a real mismatch.
        allowed = set(getattr(diffusion_module, "flow_time_parameter_names", list)()) | {
            "conditioning.t_fourier.w",
            "conditioning.t_fourier.b",
        }
        res = diffusion_module.load_state_dict(state["diffusion_module"], strict=False)
        stale = [k for k in res.missing_keys if k not in allowed]
        if stale or res.unexpected_keys:
            raise RuntimeError(
                f"checkpoint does not match the configured denoiser — missing {stale[:6]}, "
                f"unexpected {list(res.unexpected_keys)[:6]}"
            )
        if res.missing_keys and rank == 0:
            print(
                f"[train] checkpoint predates flow-time conditioning; "
                f"{len(res.missing_keys)} t-head tensor(s) freshly initialized",
                flush=True,
            )
        temp_embedder.load_state_dict(state["temp_embedder"])
        optim.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        global_step = state["global_step"]
        if rank == 0:
            print(f"[train] resumed from {ckpt_path} at step {global_step}", flush=True)

    ddp = DDP(trainable, device_ids=[local_rank]) if distributed else trainable

    dataset = make_dataset(cfg, rank=rank, world_size=world_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2, pin_memory=False)
    cond_cache = _CondCache(cfg.data.cache_dir, device)

    # Held-out stream. Built with world_size=1 and num_workers=0 so every rank
    # walks the identical frames in the identical order — the validation number is
    # then rank-independent and needs no collective.
    val_iter = None
    if cfg.val_every > 0 and cfg.data.val_domains:
        val_ds = make_dataset(cfg, rank=0, world_size=1, domains=cfg.data.val_domains)
        val_iter = _cycle(
            torch.utils.data.DataLoader(val_ds, batch_size=None, num_workers=0)
        )
        if rank == 0:
            print(
                f"[train] validation on {cfg.data.val_domains} every "
                f"{cfg.val_every} steps ({cfg.val_batches} batches)",
                flush=True,
            )
    elif rank == 0:
        print("[train] no validation (set val_every and data.val_domains)", flush=True)
    cfg_dict = config_to_dict(cfg)
    accum = cfg.optim.grad_accum
    wandb_run = _init_wandb(cfg, cfg_dict, global_step) if rank == 0 else None

    ddp.train()
    optim.zero_grad(set_to_none=True)
    data_iter = _cycle(loader)
    micro = 0
    running = 0.0
    while global_step < cfg.optim.max_steps:
        batch = next(data_iter)
        cond = cond_cache.get(batch.domain)
        gt = torch.from_numpy(batch.gt_coords).to(device)
        mask = torch.from_numpy(batch.atom_mask).to(device)

        is_accum_step = (micro + 1) % accum != 0
        sync_ctx = ddp.no_sync() if (distributed and is_accum_step) else _nullcontext()
        with sync_ctx:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not torch.float32):
                loss, metrics = ddp(cond, gt, mask, batch.temperature)
            scaled = loss / accum
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
        running += float(loss.detach())
        micro += 1

        if micro % accum == 0:
            if scaler is not None:
                scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1

            if rank == 0 and global_step % cfg.log_every == 0:
                lr = sched.get_last_lr()[0]
                avg_loss = running / (accum * cfg.log_every)
                print(f"[train] step {global_step}/{cfg.optim.max_steps} "
                      f"loss {avg_loss:.4f} "
                      f"mse {metrics['mse']:.4f} sigma {metrics['sigma_mean']:.2f} lr {lr:.2e}",
                      flush=True)
                if wandb_run is not None:
                    log_data = {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        **{f"train/{k}": v for k, v in metrics.items()},
                    }
                    wandb_run.log(log_data, step=global_step)
                running = 0.0
            # Every rank runs this (identical batches, no collective), so the
            # ranks stay in lockstep; only rank 0 reports.
            if val_iter is not None and global_step % cfg.val_every == 0:
                val = _run_validation(
                    trainable, cond_cache, val_iter, cfg.val_batches,
                    amp_dtype, device, seed=cfg.seed,
                )
                if rank == 0 and val:
                    print(f"[train] step {global_step} "
                          f"val_loss {val['loss']:.4f} val_mse {val['mse']:.4f}",
                          flush=True)
                    if wandb_run is not None:
                        wandb_run.log(
                            {f"val/{k}": v for k, v in val.items()}, step=global_step
                        )

            if rank == 0 and global_step % cfg.ckpt_every == 0:
                _save_checkpoint(ckpt_path, trainable, optim, sched, scaler, global_step, cfg_dict)

    if rank == 0:
        _save_checkpoint(ckpt_path, trainable, optim, sched, scaler, global_step, cfg_dict)
        print(f"[train] done at step {global_step}; checkpoint {ckpt_path}", flush=True)
        if wandb_run is not None:
            wandb_run.finish()
    if distributed:
        torch.distributed.destroy_process_group()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
