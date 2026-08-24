"""Temperature conditioning for the diffusion denoiser.

mdCATH samples the same domain at five temperatures; one finetuned model should
reproduce all of them. Temperature enters through the single-representation
``s_inputs`` (dim ``c_s_inputs`` = 451): the embedder maps a scalar T (K) to a
451-vector added to every token's ``s_inputs`` before the diffusion module.

The final linear layer is **zero-initialized**, so at step 0 the added bias is
exactly zero and the denoiser reproduces the pretrained model — the temperature
signal is learned from that identity starting point.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TempNorm:
    """Normalization applied to raw temperature before Fourier features."""

    ref: float = 379.0
    scale: float = 65.0

    def __call__(self, temp_kelvin):
        return (temp_kelvin - self.ref) / self.scale


def build_temperature_embedder(cfg):
    """Construct a :class:`TemperatureEmbedder` from a ``TemperatureConfig``.

    Kept as a factory so :mod:`.config` stays torch-free; import torch only here.
    """
    cls = _make_module()
    return cls(
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_fourier=cfg.num_fourier,
        norm=TempNorm(ref=cfg.ref, scale=cfg.scale),
        film=getattr(cfg, "film", False),
    )


def _make_module():
    import torch
    from torch import nn

    class _TemperatureEmbedder(nn.Module):
        """Scalar temperature (K) → additive ``s_inputs`` bias (embed_dim).

        Uses fixed random Fourier features of the normalized temperature followed
        by a 2-layer MLP whose output layer is zero-initialized (identity at init).
        """

        def __init__(self, embed_dim: int, hidden_dim: int, num_fourier: int, norm: TempNorm, film: bool = False):
            super().__init__()
            self.norm = norm
            self.film = bool(film)
            # Fixed (non-trainable) Fourier frequencies spanning several octaves,
            # so a small MLP can express smooth-to-sharp temperature dependence.
            freqs = 2.0 ** torch.linspace(-2.0, 5.0, num_fourier)
            self.register_buffer("freqs", freqs, persistent=True)
            self.mlp = nn.Sequential(
                nn.Linear(2 * num_fourier, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            # Zero-init the output layer: step-0 bias is 0 → denoiser unchanged.
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            # Multiplicative (FiLM) gain head. Separate from the bias MLP so a
            # checkpoint trained without it loads unchanged, and also zero-init so
            # the gain starts at exactly 1.0.
            if self.film:
                self.gain = nn.Sequential(
                    nn.Linear(2 * num_fourier, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
                nn.init.zeros_(self.gain[-1].weight)
                nn.init.zeros_(self.gain[-1].bias)

        def features(self, temp_kelvin):
            t = torch.as_tensor(temp_kelvin, dtype=self.freqs.dtype, device=self.freqs.device)
            t = t.reshape(-1)  # (B,)
            tn = self.norm(t)[:, None] * self.freqs[None, :]  # (B, F)
            return torch.cat([tn.sin(), tn.cos()], dim=-1)  # (B, 2F)

        def forward(self, temp_kelvin):
            """Return the additive bias ``(B, embed_dim)`` for temperatures ``(B,)``."""
            return self.mlp(self.features(temp_kelvin))

        def scale(self, temp_kelvin):
            """Return the multiplicative gain ``(B, embed_dim)``; ``1.0`` when off.

            Exactly 1.0 at init (zero-init head), so enabling ``film`` does not
            perturb a pretrained or resumed model until the gain is trained.
            """
            f = self.features(temp_kelvin)
            if not self.film:
                return torch.ones(f.shape[0], self.mlp[-1].out_features, dtype=f.dtype, device=f.device)
            return 1.0 + self.gain(f)

    return _TemperatureEmbedder


def __getattr__(name):
    # Lazily materialize the nn.Module subclass so importing this module does not
    # require torch (mirrors the lazy-heavy-import convention in ensemble.py).
    if name == "TemperatureEmbedder":
        cls = _make_module()
        globals()["TemperatureEmbedder"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
