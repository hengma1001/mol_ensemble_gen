"""A self-contained, weight-compatible reimplementation of ESMFold2's denoiser.

Why this exists
---------------
Everything in :mod:`mol_ensemble_gen.training` trains and samples *only* the
diffusion denoiser: the trunk's output is captured once per domain
(:mod:`~mol_ensemble_gen.training.featurize`) and replayed, so at train/sample
time the sole piece of ESMFold2 that runs is
``structure_head.diffusion_module`` plus two stateless geometry helpers. This
module reimplements exactly that surface from scratch, so the trainer, the losses
and the flow ODE sampler can run against our own code — no ``transformers`` or
``esm`` import, no 1.5 GB model load, and every tensor inspectable.

Weight compatibility
--------------------
Module and parameter *names* mirror the reference implementation exactly, so the
pretrained tensors load with a strict ``load_state_dict``: 345 tensors,
131,501,446 parameters. :func:`load_denoiser` does that from either an ESMFold2
checkpoint or one of our training checkpoints. :func:`state_dict_signature` is
the cheap structural check; ``tests/test_model_denoiser.py`` asserts the
signature offline and numerical parity against the real module on GPU.

Scope
-----
This is the *denoiser only* — the part that is trained and the only part needed
once conditioning is cached. The trunk (pairformer/MSA) and the ESMC language
model are **not** reimplemented here; folding a brand-new sequence still needs
the reference model to produce conditioning once. See ``DESIGN.md`` §12.

Fidelity notes (deliberate, and load-bearing for parity)
-------------------------------------------------------
* Sliding-window atom attention reproduces the reference's **non-flash** fallback
  path, including its ``bfloat16`` promotion of q/k/v, the self-attention
  diagonal forced on top of the window mask, and the zeroing of padded rows.
* 3D RoPE ``cos``/``sin`` are built in fp32 then cast to ``bfloat16``, as upstream.
* ``AttentionPairBias`` reproduces the standard (unfused) kernel path; the fused
  and cuEquivariance kernels are inference-only optimizations that the reference
  gates behind ``set_kernel_backend`` and are not modelled.
* Pair conditioning runs its transitions under ``autocast(device_type="cuda",
  bfloat16)`` exactly as upstream, which is a no-op on CPU tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# --- reference constants (modeling_esmfold2_common.py) ---------------------
CHAR_VOCAB_SIZE = 64
MAX_CHARS = 4
XYZ_DIMS = 3
MAX_ATOMIC_NUMBER = 128
#: 3 + 1 + 1 + 128 + 64*4 = 389
ATOM_FEATURE_DIM = XYZ_DIMS + 1 + 1 + MAX_ATOMIC_NUMBER + CHAR_VOCAB_SIZE * MAX_CHARS
LAYER_NORM_EPS = 1e-5


@dataclass
class DenoiserConfig:
    """Architecture hyper-parameters of ``DiffusionModule``.

    Defaults are the pretrained ``biohub/ESMFold2`` values; they determine every
    parameter shape, so a mismatch surfaces immediately as a ``load_state_dict``
    error rather than as silent numerical drift.
    """

    c_atom: int = 128
    c_token: int = 768
    c_z: int = 256
    c_s_inputs: int = 451
    sigma_data: float = 16.0
    fourier_dim: int = 256
    atom_num_blocks: int = 3
    atom_num_heads: int = 4
    token_num_blocks: int = 12
    token_num_heads: int = 16
    transition_multiplier: int = 2
    swa_window_size: int = 128
    spatial_rope_base_frequency: float = 20.0
    n_spatial_rope_pairs_per_axis: int = 2
    n_uid_rope_pairs: int = 10
    uid_rope_base_frequency: float = 10000.0

    # --- native flow-time conditioning (see DESIGN.md §13) -----------------
    #: How the denoiser is told *where on the flow path* it is.
    #:
    #: * ``"off"``     — pretrained behaviour: only the EDM noise embedding of
    #:   ``0.25·ln(σ/σ_d)``. Keeps the state dict byte-identical to the released
    #:   checkpoint, so ``load_state_dict(strict=True)`` works.
    #: * ``"add"``     — additionally embed ``t`` directly and **add** it, with the
    #:   output projection zero-initialized. At step 0 the contribution is exactly
    #:   zero, so the model still reproduces the pretrained one; native-``t``
    #:   conditioning is then learned from that identity start.
    #: * ``"replace"`` — embed ``t`` *instead of* the log-σ features. The purest
    #:   flow parameterization, but the pretrained weights never saw this input, so
    #:   identity-at-init is lost and real retraining is required.
    t_conditioning: str = "off"
    t_fourier_dim: int = 256
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t_conditioning not in ("off", "add", "replace"):
            raise ValueError(
                f"t_conditioning must be 'off', 'add' or 'replace', got {self.t_conditioning!r}"
            )


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def gather_token_to_atom(token_features: Tensor, atom_to_token_idx: Tensor) -> Tensor:
    """Broadcast ``[B, L, d]`` token features to ``[B, A, d]`` per-atom features."""
    idx = atom_to_token_idx.unsqueeze(-1).expand(-1, -1, token_features.size(-1))
    return torch.gather(token_features, 1, idx)


def scatter_atom_to_token(
    atom_features: Tensor,
    atom_to_token_idx: Tensor,
    n_tokens: int,
    atom_mask: Tensor | None = None,
) -> Tensor:
    """Mean-aggregate ``[B, A, d]`` atom features into ``[B, L, d]`` token features.

    Masked atoms are routed to a scratch row at index ``n_tokens`` which is then
    dropped, so padding never contributes to a token's mean.
    """
    b, a, d = atom_features.shape
    n_out = n_tokens
    idx = atom_to_token_idx
    if atom_mask is not None:
        idx = torch.where(atom_mask, atom_to_token_idx, n_tokens)
        n_out = n_tokens + 1
    idx_expanded = idx.unsqueeze(-1).expand(b, a, d)
    out = torch.zeros(b, n_out, d, device=atom_features.device, dtype=atom_features.dtype)
    out.scatter_reduce_(1, idx_expanded, atom_features, reduce="mean", include_self=False)
    return out[:, :n_tokens, :]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to ``[B, L, H, D]`` with per-batch ``cos``/``sin`` of ``[B, L, D/2]``."""
    ro_dim = cos.shape[-1] * 2
    cos = cos.unsqueeze(2).repeat(1, 1, 1, 2)
    sin = sin.unsqueeze(2).repeat(1, 1, 1, 2)
    return torch.cat(
        [x[..., :ro_dim] * cos + _rotate_half(x[..., :ro_dim]) * sin, x[..., ro_dim:]],
        dim=-1,
    )


def build_3d_rope(
    ref_pos: Tensor,
    ref_space_uid: Tensor,
    head_dim: int,
    n_spatial_per_axis: int = 4,
    n_uid_pairs: int = 2,
    spatial_base_freq: float = 10000.0,
    uid_base_freq: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Build ``(cos, sin)`` for spatial (x/y/z) RoPE concatenated with UID RoPE.

    The frequency block is zero-padded out to ``head_dim // 2`` so unused rotary
    channels are identity. Computed in fp32, returned in bf16 (as upstream).
    """
    device = ref_pos.device
    b, n = ref_pos.shape[:2]
    half_dim = head_dim // 2
    n_spatial_total = 3 * n_spatial_per_axis

    spatial_inv_freq = 1.0 / (
        spatial_base_freq
        ** (torch.arange(0, n_spatial_per_axis, dtype=torch.float32, device=device) / n_spatial_per_axis)
    )
    uid_inv_freq = 1.0 / (
        uid_base_freq
        ** (torch.arange(0, n_uid_pairs, dtype=torch.float32, device=device) / n_uid_pairs)
    )

    spatial_freqs = torch.einsum("bna,k->bnak", ref_pos.float(), spatial_inv_freq)
    spatial_freqs = spatial_freqs.reshape(b, n, n_spatial_total)
    uid_freqs = torch.einsum("bn,k->bnk", ref_space_uid.float(), uid_inv_freq)

    n_active = n_spatial_total + n_uid_pairs
    freqs = torch.cat([spatial_freqs, uid_freqs], dim=-1)
    if n_active < half_dim:
        pad = torch.zeros(b, n, half_dim - n_active, device=device, dtype=torch.float32)
        freqs = torch.cat([freqs, pad], dim=-1)

    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


def qk_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)


class TransitionLayer(nn.Module):
    """SwiGLU transition: LayerNorm → (a_proj, b_proj) → silu(a)·b → out_proj."""

    def __init__(self, d_model: int, n: int, eps: float = LAYER_NORM_EPS) -> None:
        super().__init__()
        hidden = n * d_model
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.a_proj = nn.Linear(d_model, hidden, bias=False)
        self.b_proj = nn.Linear(d_model, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.out_proj(F.silu(self.a_proj(x)) * self.b_proj(x))


class AdaptiveLayerNorm(nn.Module):
    """adaLN-Zero: gate·LN(a) + shift, both predicted from the conditioning ``s``."""

    def __init__(self, d_model: int, d_cond: int, eps: float = LAYER_NORM_EPS) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_cond = d_cond
        self.eps = eps
        self.s_scale = nn.Parameter(torch.ones(d_cond))
        self.s_gate = nn.Linear(d_cond, d_model, bias=True)
        self.s_shift = nn.Linear(d_cond, d_model, bias=False)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        a_norm = F.layer_norm(a, (self.d_model,), None, None, self.eps)
        s_norm = F.layer_norm(s, (self.d_cond,), self.s_scale, None, self.eps)
        return torch.sigmoid(self.s_gate(s_norm)) * a_norm + self.s_shift(s_norm)


class FourierEmbedding(nn.Module):
    """``cos(2π(t·w + b))`` with fixed random ``w``/``b`` (buffers, so they load)."""

    w: Tensor
    b: Tensor

    def __init__(self, c: int) -> None:
        super().__init__()
        self.c = c
        self.register_buffer("w", torch.randn(c))
        self.register_buffer("b", torch.randn(c))

    def forward(self, t_hat: Tensor) -> Tensor:
        t = torch.as_tensor(t_hat, device=self.w.device, dtype=self.w.dtype).reshape(-1)
        return torch.cos(2.0 * math.pi * (t[:, None] * self.w[None, :] + self.b[None, :]))


class SwiGLUFFN(nn.Module):
    """Atom-block FFN; hidden size rounded up to a multiple of 256 as upstream."""

    def __init__(self, d_model: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        hidden_size = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.w_up = nn.Linear(d_model, 2 * hidden_size, bias=False)
        self.w_down = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.w_up.weight.dtype)
        x1, x2 = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(x1) * x2)


# ---------------------------------------------------------------------------
# sliding-window atom attention
# ---------------------------------------------------------------------------


class SWA3DRoPEAttention(nn.Module):
    """Gated sliding-window self-attention over atoms, with 3D RoPE.

    Mirrors the reference's fallback (no flash-attn) kernel: q/k/v are promoted to
    bf16, the window is measured in *valid-atom rank* rather than raw position so
    padding does not consume window budget, the diagonal is always attendable, and
    padded rows are zeroed afterwards.
    """

    def __init__(self, d_model: int, n_heads: int, half_window: int = 64) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.half_window = half_window

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, attention_params: tuple) -> Tensor:
        b, n = x.shape[:2]
        cos, sin = attention_params[0], attention_params[1]

        x_input = x
        qkv = self.Wqkv(x).view(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        q, k = qk_norm(q), qk_norm(k)
        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)

        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        if len(attention_params) > 2:
            valid = torch.zeros(b * n, dtype=torch.bool, device=q.device)
            valid[attention_params[2]] = True
            valid = valid.view(b, n)
        else:
            valid = torch.ones(b, n, dtype=torch.bool, device=q.device)

        rank = torch.cumsum(valid, dim=1) - 1
        within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= self.half_window
        allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
        allowed |= torch.eye(n, dtype=torch.bool, device=q.device)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed.unsqueeze(1),
            scale=self.scale,
        ).transpose(1, 2)
        out = out * valid.unsqueeze(-1).unsqueeze(-1)

        out = out.to(input_dtype).reshape(b, n, -1)
        out = out * torch.sigmoid(self.gate_proj(x_input))
        return self.out_proj(out)


class SWAAtomBlock(nn.Module):
    """adaLN-Zero + sliding-window attention + SwiGLU FFN, over atoms."""

    def __init__(
        self,
        d_atom: int,
        n_heads: int,
        half_window: int = 64,
        expansion_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_atom, elementwise_affine=False)
        self.ffn_norm = nn.RMSNorm(d_atom, elementwise_affine=False)
        adaln_linear = nn.Linear(d_atom, 6 * d_atom, bias=False)
        nn.init.zeros_(adaln_linear.weight)
        # Sequential so the parameter key is `adaln_modulation.1.weight`, as upstream.
        self.adaln_modulation = nn.Sequential(nn.SiLU(), adaln_linear)
        self.attn = SWA3DRoPEAttention(d_atom, n_heads, half_window=half_window)
        self.ffn = SwiGLUFFN(d_atom, expansion_ratio)

    @staticmethod
    def _rms_adaln(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],)) * (1 + scale) + shift

    def forward(self, x: Tensor, c_l: Tensor, attention_params: tuple) -> Tensor:
        mod = self.adaln_modulation(c_l)
        if mod.dim() == 2:
            mod = mod.unsqueeze(1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.chunk(6, dim=-1)

        x = x + gate_a * self.attn(self._rms_adaln(x, scale_a, shift_a), attention_params)
        x = x + gate_f * self.ffn(self._rms_adaln(x, scale_f, shift_f))
        return x


class SWAAtomTransformer(nn.Module):
    """Stack of :class:`SWAAtomBlock`, plus the 3D-RoPE builder for its head dim."""

    def __init__(
        self,
        d_atom: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.swa_window_size = swa_window_size
        self.head_dim = d_atom // n_heads
        self.spatial_rope_base_frequency = spatial_rope_base_frequency
        self.n_spatial_rope_pairs_per_axis = n_spatial_rope_pairs_per_axis
        self.n_uid_rope_pairs = n_uid_rope_pairs
        self.uid_rope_base_frequency = uid_rope_base_frequency
        self.blocks = nn.ModuleList(
            [
                SWAAtomBlock(
                    d_atom=d_atom,
                    n_heads=n_heads,
                    half_window=swa_window_size // 2,
                    expansion_ratio=expansion_ratio,
                )
                for _ in range(n_blocks)
            ]
        )

    def build_3d_rope(self, ref_pos: Tensor, ref_space_uid: Tensor) -> tuple[Tensor, Tensor]:
        return build_3d_rope(
            ref_pos=ref_pos,
            ref_space_uid=ref_space_uid,
            head_dim=self.head_dim,
            n_spatial_per_axis=self.n_spatial_rope_pairs_per_axis,
            n_uid_pairs=self.n_uid_rope_pairs,
            spatial_base_freq=self.spatial_rope_base_frequency,
            uid_base_freq=self.uid_rope_base_frequency,
        )

    def _build_3d_rope(self, ref_pos: Tensor, ref_space_uid: Tensor) -> tuple[Tensor, Tensor]:
        """Underscore alias — the name upstream code calls."""
        return self.build_3d_rope(ref_pos, ref_space_uid)

    def forward(
        self,
        q_l: Tensor,
        c_l: Tensor,
        attention_params: tuple,
        return_intermediates: bool = False,
    ):
        intermediates: list[Tensor] = []
        for block in self.blocks:
            q_l = block(q_l, c_l, attention_params)
            if return_intermediates:
                intermediates.append(q_l)
        if return_intermediates:
            return q_l, intermediates
        return q_l


class ESMFold2AtomEncoder(nn.Module):
    """Embed reference-conformer atom features (+ noisy coords) and pool to tokens."""

    def __init__(
        self,
        d_atom: int = 128,
        d_token: int = 768,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        structure_prediction: bool = True,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_atom = d_atom
        self.d_token = d_token
        self.structure_prediction = structure_prediction

        self.atom_linear = nn.Linear(ATOM_FEATURE_DIM, d_atom, bias=False)
        self.atom_norm = nn.LayerNorm(d_atom)
        if structure_prediction:
            # 6 = noisy coords (3) ++ previous prediction (3)
            self.coords_linear = nn.Linear(6, d_atom, bias=False)

        self.atom_transformer = SWAAtomTransformer(
            d_atom=d_atom,
            n_blocks=n_blocks,
            n_heads=n_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=expansion_ratio,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )
        out_dim = d_token if structure_prediction else d_token // 2
        self.atom_to_token_linear = nn.Linear(d_atom, out_dim, bias=False)

    def forward(
        self,
        ref_pos: Tensor,
        atom_attention_mask: Tensor,
        ref_space_uid: Tensor,
        ref_charge: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        atom_to_token: Tensor,
        r_l: Tensor | None = None,
        pred_r1: Tensor | None = None,
        s_i: Tensor | None = None,
        z_ij: Tensor | None = None,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
        inference_cache: dict | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple, list[Tensor]]:
        """Return ``(a, q, c, attention_params, intermediates)``.

        ``inference_cache`` memoizes the step-invariant pieces (feature embedding,
        RoPE tables, attention bookkeeping) across diffusion steps.
        """
        b, n = ref_pos.shape[:2]

        layer_cache = None
        if inference_cache is not None:
            layer_cache = inference_cache.setdefault("atomencoder", {})

        if layer_cache is None or len(layer_cache) == 0:
            atom_feats = torch.cat(
                [
                    ref_pos,
                    ref_charge.unsqueeze(-1),
                    atom_attention_mask.unsqueeze(-1),
                    ref_element,
                    ref_atom_name_chars.reshape(b, n, MAX_CHARS * CHAR_VOCAB_SIZE),
                ],
                dim=-1,
            )
            c_base = self.atom_norm(self.atom_linear(atom_feats))
            cos, sin = self.atom_transformer.build_3d_rope(ref_pos, ref_space_uid)
            cos = cos.repeat_interleave(num_diffusion_samples, 0)
            sin = sin.repeat_interleave(num_diffusion_samples, 0)
            mask_exp = atom_attention_mask.repeat_interleave(num_diffusion_samples, 0)
            seqlens = mask_exp.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(mask_exp.flatten(), as_tuple=False).flatten()
            max_seqlen = int(seqlens.max().item())
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
            attention_params = (cos, sin, indices, cu_seqlens, max_seqlen)
            n_tokens = int(atom_to_token.max().item()) + 1
            if layer_cache is not None:
                layer_cache["c_base"] = c_base
                layer_cache["attention_params"] = attention_params
                layer_cache["mask_exp"] = mask_exp
                layer_cache["n_tokens"] = n_tokens
                layer_cache["atom_to_token_exp"] = atom_to_token.repeat_interleave(
                    num_diffusion_samples, 0
                )
        else:
            c_base = layer_cache["c_base"]
            attention_params = layer_cache["attention_params"]
            mask_exp = layer_cache["mask_exp"]
            n_tokens = layer_cache["n_tokens"]

        c = c_base
        q = c
        if self.structure_prediction and r_l is not None:
            q = q.repeat_interleave(num_diffusion_samples, 0)
            if pred_r1 is None:
                pred_r1 = torch.zeros_like(r_l)
            q = q + self.coords_linear(torch.cat([r_l, pred_r1], dim=-1))
        c = c.repeat_interleave(num_diffusion_samples, 0)

        result = self.atom_transformer(
            q_l=q,
            c_l=c,
            attention_params=attention_params,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            q, intermediates = result
        else:
            q, intermediates = result, []

        q_to_a = F.relu(self.atom_to_token_linear(q))
        if layer_cache is not None and "atom_to_token_exp" in layer_cache:
            atom_to_token_exp = layer_cache["atom_to_token_exp"]
        else:
            atom_to_token_exp = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
        a = scatter_atom_to_token(q_to_a, atom_to_token_exp, n_tokens, atom_mask=mask_exp.bool())
        return a, q, c, attention_params, intermediates


class ESMFold2AtomDecoder(nn.Module):
    """Broadcast token features back to atoms and read out a coordinate update."""

    def __init__(
        self,
        d_atom: int = 128,
        d_token: int = 768,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.token_to_atom_linear = nn.Linear(d_token, d_atom, bias=False)
        self.atom_transformer = SWAAtomTransformer(
            d_atom=d_atom,
            n_blocks=n_blocks,
            n_heads=n_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=expansion_ratio,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )
        self.norm = nn.LayerNorm(d_atom)
        self.output_linear = nn.Linear(d_atom, XYZ_DIMS, bias=False)

    def forward(
        self,
        a_i: Tensor,
        q_l: Tensor,
        c_l: Tensor,
        p_lm: tuple,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        atom_to_token_exp = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
        a_to_q = gather_token_to_atom(self.token_to_atom_linear(a_i), atom_to_token_exp)
        q_l = q_l + a_to_q

        result = self.atom_transformer(
            q_l=q_l,
            c_l=c_l,
            attention_params=p_lm,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            q_l, intermediates = result
        else:
            q_l, intermediates = result, []

        return self.output_linear(self.norm(q_l)), intermediates


# ---------------------------------------------------------------------------
# token transformer
# ---------------------------------------------------------------------------


class AttentionPairBias(nn.Module):
    """Gated multi-head attention biased by the pair representation ``z``."""

    def __init__(
        self,
        d_model: int,
        d_pair: int,
        num_heads: int,
        d_cond: int | None = None,
        use_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        d_cond = d_cond or d_model

        if use_conditioning:
            self.adaln = AdaptiveLayerNorm(d_model, d_cond, eps=LAYER_NORM_EPS)
            self.out_gate = nn.Linear(d_cond, d_model, bias=True)
            nn.init.zeros_(self.out_gate.weight)
            nn.init.constant_(self.out_gate.bias, -2.0)
        else:
            self.pre_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if d_pair > 0:
            self.pair_norm = nn.LayerNorm(d_pair, eps=LAYER_NORM_EPS)
            self.pair_bias_proj = nn.Linear(d_pair, num_heads, bias=False)

    def forward(
        self,
        a: Tensor,
        s: Tensor | None,
        z: Tensor,
        beta: Tensor | float = 0.0,
        attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
    ) -> Tensor:
        bsz, n_queries, d_model = a.shape
        x = self.adaln(a, s) if s is not None else self.pre_norm(a)

        n_keys = x.shape[1]
        q = self.q_proj(x).view(bsz, n_queries, self.num_heads, self.head_dim)
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        k = k.view(bsz, n_keys, self.num_heads, self.head_dim)
        v = v.view(bsz, n_keys, self.num_heads, self.head_dim)

        # Batch-1 conditioning is broadcast to the diffusion-sample batch here, so
        # z is never materialized per sample upstream of this call.
        if z.dim() == 4 and z.shape[0] != bsz and num_diffusion_samples > 1:
            z = z.repeat_interleave(num_diffusion_samples, dim=0)
        if (
            attention_mask is not None
            and attention_mask.shape[0] != bsz
            and num_diffusion_samples > 1
        ):
            attention_mask = attention_mask.repeat_interleave(num_diffusion_samples, dim=0)

        g = torch.sigmoid(self.g_proj(x)).view(bsz, n_queries, self.num_heads, self.head_dim)
        logits = torch.einsum("... i h d, ... j h d -> ... i j h", q, k) * self.scale
        pair_bias = self.pair_bias_proj(self.pair_norm(z)) if z.dim() == 4 else z.unsqueeze(-1)
        logits = logits + pair_bias.to(dtype=logits.dtype)

        if attention_mask is not None:
            min_val = torch.finfo(logits.dtype).min
            mask_bias = torch.where(attention_mask.bool()[:, None, :, None], 0.0, min_val)
            logits = logits + mask_bias.to(dtype=logits.dtype)

        # Softmax over keys (dim -2 with the [..., i, j, h] layout).
        attn = torch.softmax(logits, dim=-2).to(dtype=v.dtype)
        ctx = g * torch.einsum("... i j h, ... j h d -> ... i h d", attn, v)
        out = self.out_proj(ctx.reshape(bsz, n_queries, d_model))
        if s is not None:
            out = torch.sigmoid(self.out_gate(s)) * out
        return out


class ConditionedTransitionBlock(nn.Module):
    """SwiGLU transition with adaLN conditioning and a zero-init output gate."""

    def __init__(
        self,
        d_model: int,
        d_cond: int | None = None,
        transition_multiplier: int = 2,
        use_conditioning: bool = True,
    ) -> None:
        super().__init__()
        d_cond = d_cond or d_model
        hidden = transition_multiplier * d_model

        if use_conditioning:
            self.adaln = AdaptiveLayerNorm(d_model, d_cond, eps=LAYER_NORM_EPS)
            self.output_gate = nn.Linear(d_cond, d_model, bias=True)
            nn.init.zeros_(self.output_gate.weight)
            nn.init.constant_(self.output_gate.bias, -2.0)
        else:
            self.pre_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

        self.lin_swish = nn.Linear(d_model, 2 * hidden, bias=False)
        self.lin_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, a: Tensor, s: Tensor | None) -> Tensor:
        x = self.adaln(a, s) if s is not None else self.pre_norm(a)
        swish_a, swish_b = self.lin_swish(x).chunk(2, dim=-1)
        out = self.lin_out(F.silu(swish_a) * swish_b)
        if s is not None:
            out = torch.sigmoid(self.output_gate(s)) * out
        return out


class DiffusionTransformer(nn.Module):
    """Interleaved pair-biased attention and conditioned transition blocks."""

    def __init__(
        self,
        d_model: int,
        d_pair: int,
        num_heads: int,
        num_blocks: int,
        d_cond: int | None = None,
        transition_multiplier: int = 2,
        use_conditioning: bool = True,
    ) -> None:
        super().__init__()
        d_cond = d_cond or d_model
        self.attn_blocks = nn.ModuleList(
            [
                AttentionPairBias(
                    d_model=d_model,
                    d_pair=d_pair,
                    num_heads=num_heads,
                    d_cond=d_cond,
                    use_conditioning=use_conditioning,
                )
                for _ in range(num_blocks)
            ]
        )
        self.transition_blocks = nn.ModuleList(
            [
                ConditionedTransitionBlock(
                    d_model=d_model,
                    d_cond=d_cond,
                    transition_multiplier=transition_multiplier,
                    use_conditioning=use_conditioning,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        a: Tensor,
        s: Tensor | None,
        z: Tensor,
        beta: Tensor | float = 0.0,
        attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        intermediates: list[Tensor] = []
        x = a
        for attn, transition in zip(self.attn_blocks, self.transition_blocks):
            x = x + attn(
                x, s, z, beta,
                attention_mask=attention_mask,
                num_diffusion_samples=num_diffusion_samples,
            )
            x = x + transition(x, s)
            if return_intermediates:
                intermediates.append(x)
        return x, intermediates


class DiffusionConditioning(nn.Module):
    """Build the noise-conditioned single ``s`` and pair ``z`` representations."""

    def __init__(
        self,
        c_z: int = 256,
        c_s: int = 768,
        c_s_inputs: int = 451,
        sigma_data: float = 16.0,
        fourier_dim: int = 256,
        transition_multiplier: int = 2,
        layer_norm_eps: float = LAYER_NORM_EPS,
        t_conditioning: str = "off",
        t_fourier_dim: int = 256,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        self.t_conditioning = t_conditioning

        self.z_input_norm = nn.LayerNorm(2 * c_z, eps=layer_norm_eps)
        self.z_proj = nn.Linear(2 * c_z, c_z, bias=False)
        self.z_transitions = nn.ModuleList(
            [TransitionLayer(c_z, n=transition_multiplier, eps=layer_norm_eps) for _ in range(2)]
        )

        self.s_input_norm = nn.LayerNorm(c_s_inputs, eps=layer_norm_eps)
        self.s_proj = nn.Linear(c_s_inputs, c_s, bias=False)
        self.fourier = FourierEmbedding(fourier_dim)
        self.noise_norm = nn.LayerNorm(fourier_dim, eps=layer_norm_eps)
        self.noise_proj = nn.Linear(fourier_dim, c_s, bias=False)
        self.s_transitions = nn.ModuleList(
            [TransitionLayer(c_s, n=transition_multiplier, eps=layer_norm_eps) for _ in range(2)]
        )

        # Native flow-time head. Only built when asked, so the default state dict
        # stays identical to the released checkpoint.
        if t_conditioning in ("add", "replace"):
            self.t_fourier = FourierEmbedding(t_fourier_dim)
            self.t_norm = nn.LayerNorm(t_fourier_dim, eps=layer_norm_eps)
            self.t_proj = nn.Linear(t_fourier_dim, c_s, bias=False)
            if t_conditioning == "add":
                # Zero-init: contributes exactly nothing at step 0, so the model
                # still reproduces the pretrained one and learns t from there.
                nn.init.zeros_(self.t_proj.weight)

    def forward(
        self,
        t_hat: Tensor,
        s_inputs: Tensor,
        s_trunk: Tensor | None,
        z_trunk: Tensor,
        relative_position_encoding: Tensor,
        sigma_data: float | None = None,
        num_diffusion_samples: int = 1,
        inference_cache: dict[str, Tensor] | None = None,
        flow_t: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        sigma = self.sigma_data if sigma_data is None else float(sigma_data)
        target_batch = z_trunk.shape[0] * num_diffusion_samples

        # z is independent of t, so it is cached across diffusion steps.
        if inference_cache is not None and "z" in inference_cache:
            z = inference_cache["z"]
        else:
            z_rel = relative_position_encoding.to(dtype=torch.float32)
            z = torch.cat([z_trunk.to(dtype=torch.float32), z_rel], dim=-1)
            z = self.z_proj(self.z_input_norm(z))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for block in self.z_transitions:
                    z = z + block(z)
            if inference_cache is not None:
                inference_cache["z"] = z

        s_inputs_eff = s_inputs
        if s_inputs_eff.shape[0] != target_batch:
            s_inputs_eff = s_inputs_eff.repeat_interleave(num_diffusion_samples, 0)
        s = self.s_proj(self.s_input_norm(s_inputs_eff.to(dtype=torch.float32)))

        t = torch.as_tensor(t_hat, dtype=torch.float32, device=s.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(target_batch)
        elif t.shape[0] != target_batch:
            t = t.repeat_interleave(num_diffusion_samples, 0)
        if self.t_conditioning != "replace":
            t_noise = 0.25 * torch.log((t / sigma).clamp(min=1e-20))
            n = self.noise_proj(self.noise_norm(self.fourier(t_noise)))
            s = s + n.unsqueeze(1)

        if self.t_conditioning in ("add", "replace"):
            if flow_t is None:
                # σ carries the same information, so recover t rather than fail —
                # this keeps EDM-style callers working against a t-conditioned model.
                flow_t = t / (1.0 + t)
            ft = torch.as_tensor(flow_t, dtype=torch.float32, device=s.device).reshape(-1)
            if ft.numel() == 1:
                ft = ft.expand(target_batch)
            elif ft.shape[0] != target_batch:
                ft = ft.repeat_interleave(num_diffusion_samples, 0)
            # Centre t about the path midpoint so the Fourier features see a
            # roughly zero-mean input, as the log-σ features do.
            t_feat = self.t_proj(self.t_norm(self.t_fourier(2.0 * ft - 1.0)))
            s = s + t_feat.unsqueeze(1)

        for block in self.s_transitions:
            s = s + block(s)
        return s, z


# ---------------------------------------------------------------------------
# the denoiser
# ---------------------------------------------------------------------------


class DiffusionModule(nn.Module):
    """EDM x₀-predictor ``D(x; σ)`` — ESMFold2's all-atom diffusion denoiser.

    Preconditioning is built in: the input is scaled by ``1/√(t²+σ_d²)`` and the
    output mixes the noisy input with the network's update using the standard EDM
    skip/out coefficients, so the module maps a noisy structure directly to a
    predicted clean structure.
    """

    def __init__(self, config: DenoiserConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or DenoiserConfig(**overrides)
        self.config = cfg
        self.sigma_data = float(cfg.sigma_data)

        self.conditioning = DiffusionConditioning(
            c_z=cfg.c_z,
            c_s=cfg.c_token,
            c_s_inputs=cfg.c_s_inputs,
            sigma_data=cfg.sigma_data,
            fourier_dim=cfg.fourier_dim,
            transition_multiplier=cfg.transition_multiplier,
            t_conditioning=cfg.t_conditioning,
            t_fourier_dim=cfg.t_fourier_dim,
        )
        self.atom_encoder = ESMFold2AtomEncoder(
            d_atom=cfg.c_atom,
            d_token=cfg.c_token,
            n_blocks=cfg.atom_num_blocks,
            n_heads=cfg.atom_num_heads,
            swa_window_size=cfg.swa_window_size,
            expansion_ratio=2,
            structure_prediction=True,
            spatial_rope_base_frequency=cfg.spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=cfg.n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=cfg.n_uid_rope_pairs,
            uid_rope_base_frequency=cfg.uid_rope_base_frequency,
        )
        self.atom_decoder = ESMFold2AtomDecoder(
            d_atom=cfg.c_atom,
            d_token=cfg.c_token,
            n_blocks=cfg.atom_num_blocks,
            n_heads=cfg.atom_num_heads,
            swa_window_size=cfg.swa_window_size,
            expansion_ratio=2,
            spatial_rope_base_frequency=cfg.spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=cfg.n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=cfg.n_uid_rope_pairs,
            uid_rope_base_frequency=cfg.uid_rope_base_frequency,
        )
        self.s_to_token = nn.Linear(cfg.c_token, cfg.c_token, bias=False)
        nn.init.zeros_(self.s_to_token.weight)
        self.token_transformer = DiffusionTransformer(
            d_model=cfg.c_token,
            d_pair=cfg.c_z,
            num_heads=cfg.token_num_heads,
            num_blocks=cfg.token_num_blocks,
            d_cond=cfg.c_token,
            transition_multiplier=cfg.transition_multiplier,
            use_conditioning=True,
        )
        self.s_step_norm = nn.LayerNorm(cfg.c_token)
        self.token_norm = nn.LayerNorm(cfg.c_token)

    #: Lets callers (e.g. the loss) detect the native flow-time input without
    #: signature introspection; the reference module lacks the attribute.
    supports_flow_time = True

    @property
    def t_conditioning(self) -> str:
        return self.config.t_conditioning

    def flow_time_parameter_names(self) -> list[str]:
        """Parameter names belonging to the native flow-time head (may be empty)."""
        return [n for n, _ in self.named_parameters() if n.startswith("conditioning.t_")]

    def forward(
        self,
        x_noisy: Tensor,
        t_hat: Tensor,
        ref_pos: Tensor,
        ref_charge: Tensor,
        ref_mask: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        tok_idx: Tensor,
        s_inputs: Tensor,
        s_trunk: Tensor | None,
        z_trunk: Tensor,
        relative_position_encoding: Tensor,
        asym_id: Tensor,
        residue_index: Tensor,
        entity_id: Tensor,
        token_index: Tensor,
        sym_id: Tensor,
        sigma_data: float | None = None,
        token_attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        return_token_repr: bool = False,
        return_atom_repr: bool = False,
        inference_cache: dict[str, Tensor] | None = None,
        flow_t: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Denoise ``x_noisy`` at noise level ``t_hat``.

        Returns ``{"x_denoised", "token_repr", "atom_intermediates"}``. The
        ``asym_id``/``residue_index``/``entity_id``/``token_index``/``sym_id``
        arguments are part of the reference signature and accepted for drop-in
        compatibility; the denoiser reads position information from the cached
        ``relative_position_encoding`` instead.

        ``flow_t`` is the rectified-flow path time, used only when the config's
        ``t_conditioning`` is ``"add"``/``"replace"``. When omitted it is recovered
        from ``t_hat`` as ``σ/(1+σ)``, which carries the same information.
        """
        bsz = x_noisy.shape[0]
        sigma = self.sigma_data if sigma_data is None else float(sigma_data)
        t = torch.as_tensor(t_hat, dtype=torch.float32, device=x_noisy.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(bsz)

        s, z = self.conditioning(
            t_hat=t,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            sigma_data=sigma,
            num_diffusion_samples=num_diffusion_samples,
            inference_cache=inference_cache,
            flow_t=flow_t,
        )

        # EDM input preconditioning.
        r_noisy = x_noisy / torch.sqrt(t * t + sigma * sigma)[:, None, None]

        a, q_skip, c_skip, p_skip, enc_intermediates = self.atom_encoder(
            ref_pos=ref_pos,
            atom_attention_mask=ref_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            atom_to_token=tok_idx,
            r_l=r_noisy,
            s_i=s_trunk,
            num_diffusion_samples=num_diffusion_samples,
            return_intermediates=return_atom_repr,
            inference_cache=inference_cache,
        )

        a = a + self.s_to_token(self.s_step_norm(s))
        a, _ = self.token_transformer(
            a, s, z, beta=0.0,
            attention_mask=token_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
        )
        a = self.token_norm(a)

        r_update, dec_intermediates = self.atom_decoder(
            a_i=a,
            q_l=q_skip,
            c_l=c_skip,
            p_lm=p_skip,
            atom_to_token=tok_idx,
            atom_attention_mask=ref_mask,
            num_diffusion_samples=num_diffusion_samples,
            return_intermediates=return_atom_repr,
        )

        # EDM output preconditioning: skip-scaled input + out-scaled update.
        sigma2, t2 = sigma * sigma, t * t
        out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
        out = out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update

        atom_intermediates: Tensor | None = None
        if return_atom_repr:
            all_ints = enc_intermediates + dec_intermediates
            if all_ints:
                atom_intermediates = torch.stack(all_ints, dim=2)

        return {
            "x_denoised": out,
            "token_repr": a if return_token_repr else None,
            "atom_intermediates": atom_intermediates,
        }


# ---------------------------------------------------------------------------
# geometry helpers (DiffusionStructureHead surface used by loss/sampling)
# ---------------------------------------------------------------------------


class GeometryOps(nn.Module):
    """The two stateless geometry helpers the loss and samplers need.

    Kept as a tiny module (no parameters) so it can stand in for the reference
    ``structure_head`` wherever only ``_center_random_augmentation`` and
    ``_weighted_rigid_align`` are used — which is everywhere in
    :mod:`mol_ensemble_gen.training`.
    """

    @staticmethod
    def _random_rotations(
        n: int, dtype: torch.dtype, device: torch.device, generator=None
    ) -> Tensor:
        """Uniform random rotations via normalized quaternions (sign-fixed).

        ``generator`` is optional so training keeps drawing from the global RNG
        (unchanged behavior); validation passes one to make the draw reproducible.
        """
        q = torch.randn((n, 4), dtype=dtype, device=device, generator=generator)
        scale = torch.sqrt((q * q).sum(dim=1))
        signs = torch.where(q[:, 0] < 0, -scale, scale)
        q = q / signs[:, None]
        r, i, j, k = torch.unbind(q, dim=-1)
        two_s = 2.0 / (q * q).sum(dim=-1)
        return torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            dim=-1,
        ).reshape(n, 3, 3)

    def _center_random_augmentation(
        self, x: Tensor, atom_mask: Tensor, second_coords: Tensor | None = None,
        generator=None,
    ) -> tuple[Tensor, Tensor | None]:
        """Mask-aware centering, then a random rotation and translation.

        Pass ``generator`` to make the augmentation reproducible. Validation must
        do so: with the draw coming from the global RNG, two evaluations of the
        *same weights* score differently-oriented ground truth and land 2-2.4%
        apart, which is the same order as the training improvements being
        measured. Training leaves it ``None`` and keeps the global stream.
        """
        bsz = x.shape[0]
        mask = atom_mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (x * mask).sum(dim=1, keepdim=True) / denom
        x = x - mean
        if second_coords is not None:
            second_coords = second_coords - mean

        r = self._random_rotations(bsz, x.dtype, x.device, generator=generator)
        x = torch.einsum("bmd,bds->bms", x, r)
        if second_coords is not None:
            second_coords = torch.einsum("bmd,bds->bms", second_coords, r)

        # randn_like takes no generator, so spell the shape out — otherwise the
        # translation silently stays on the global RNG and seeding the rotation
        # alone is not enough to make the augmentation reproducible.
        t = torch.randn(
            x[:, 0:1, :].shape, dtype=x.dtype, device=x.device, generator=generator
        )
        x = x + t
        if second_coords is not None:
            second_coords = second_coords + t
        return x, second_coords

    @staticmethod
    def _weighted_rigid_align(x: Tensor, x_gt: Tensor, w: Tensor, mask: Tensor) -> Tensor:
        """Weighted Kabsch: rigidly move ``x`` onto ``x_gt``.

        The SVD and determinant run in fp32 — neither has a bf16 kernel, so
        callers must disable autocast around this (see ``training/loss.py``).
        """
        w = (mask * w).unsqueeze(-1)
        denom = w.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        mu = (x * w).sum(dim=-2, keepdim=True) / denom
        mu_gt = (x_gt * w).sum(dim=-2, keepdim=True) / denom
        x_c = x - mu
        xgt_c = x_gt - mu_gt
        h = torch.einsum("bni,bnj->bij", w * xgt_c, x_c)
        h32 = h.float()
        u, _, vh = torch.linalg.svd(h32, driver="gesvd" if h32.is_cuda else None)
        det = torch.linalg.det(u @ vh)
        ones = torch.ones_like(det)
        rot = (u @ torch.diag_embed(torch.stack([ones, ones, det], dim=-1)) @ vh).to(h.dtype)
        return x_c @ rot.transpose(-1, -2) + mu_gt


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

#: Tensor count / parameter count of the pretrained ``biohub/ESMFold2`` denoiser.
PRETRAINED_NUM_TENSORS = 345
PRETRAINED_NUM_PARAMS = 131_501_446


def state_dict_signature(state: dict) -> tuple[int, int]:
    """``(n_tensors, n_params)`` — the cheap structural fingerprint of a denoiser."""
    return len(state), sum(v.numel() for v in state.values())


def load_denoiser(
    source: str | dict,
    *,
    config: DenoiserConfig | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> DiffusionModule:
    """Build a :class:`DiffusionModule` and load pretrained/finetuned weights.

    ``source`` is either a state dict of the denoiser itself, or a path to a
    checkpoint. Our training checkpoints nest it under ``"diffusion_module"``;
    a bare denoiser state dict is also accepted.

    When the config enables a native flow-time head, that head does not exist in a
    pretrained ESMFold2 checkpoint. Those keys — and *only* those — are allowed to
    be missing; anything else missing or unexpected is still an error, so a genuine
    mismatch cannot hide behind the exemption.
    """
    if isinstance(source, dict):
        state = source
    else:
        blob = torch.load(source, map_location="cpu", weights_only=False)
        state = blob.get("diffusion_module", blob) if isinstance(blob, dict) else blob
    if "diffusion_module" in state and isinstance(state["diffusion_module"], dict):
        state = state["diffusion_module"]

    model = DiffusionModule(config)
    result = model.load_state_dict(state, strict=False)
    if strict:
        allowed = set(model.flow_time_parameter_names()) | {
            "conditioning.t_fourier.w",
            "conditioning.t_fourier.b",
        }
        unexpected_missing = [k for k in result.missing_keys if k not in allowed]
        if unexpected_missing or result.unexpected_keys:
            raise RuntimeError(
                "denoiser state dict mismatch — "
                f"missing: {unexpected_missing[:8]}, unexpected: {list(result.unexpected_keys)[:8]}"
            )
        if result.missing_keys:
            print(
                f"[denoiser] {len(result.missing_keys)} flow-time head tensor(s) "
                f"freshly initialized (t_conditioning={model.config.t_conditioning!r})"
            )
    return model.to(device)


def augment_with_generator(head, x0, mask, generator):
    """Center + randomly rotate/translate ``x0``, seeding the draw when possible.

    The ``generator`` has to reach the augmentation, not just the ``eps`` draw:
    with the rotation and translation coming off the global RNG, two validation
    passes over the *same weights and same frames* scored 2-4% apart (measured:
    40 passes spanned 2.935-3.053), which is the same magnitude as the training
    effects the metric was being used to compare.

    Only :class:`~mol_ensemble_gen.model.denoiser.GeometryOps` accepts the kwarg;
    the reference backend's ``DiffusionStructureHead`` takes
    ``(x, atom_mask, second_coords)`` and nothing else, so it is called unseeded
    and its validation keeps the old few-percent jitter. Detecting support beats
    hard-coding it: the reference signature is upstream's to change.
    """
    import inspect

    support = getattr(head, "_accepts_augmentation_generator", None)
    if support is None:
        try:
            params = inspect.signature(head._center_random_augmentation).parameters
            support = "generator" in params
        except (AttributeError, TypeError, ValueError):
            support = False
        try:
            head._accepts_augmentation_generator = support
        except AttributeError:                      # pragma: no cover - exotic heads
            pass
    if support:
        return head._center_random_augmentation(
            x0, mask, second_coords=None, generator=generator
        )
    return head._center_random_augmentation(x0, mask, second_coords=None)
