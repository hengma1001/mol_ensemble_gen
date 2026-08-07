"""Finetuning ESMFold2's diffusion module to reproduce mdCATH MD ensembles.

The stock package (:mod:`mol_ensemble_gen.ensemble`) samples a *frozen* ESMFold2
to draw configuration ensembles from the model's folding distribution. This
subpackage instead **finetunes** the diffusion denoiser so that sampling
reproduces the conformational ensembles seen in classical-FF MD (the mdCATH
dataset), conditioned on temperature.

The trunk (pair/single conditioning) is frozen and its output is a pure function
of sequence — so it is cached once per domain and reused across every frame and
temperature. Only ``model.structure_head.diffusion_module`` and a new
:class:`~mol_ensemble_gen.training.conditioning.TemperatureEmbedder` are trained.

Heavy imports (torch, h5py, esm/transformers) are done lazily inside functions so
that the light modules (:mod:`.config`, :mod:`.atom_map`) import without a GPU.
"""

from __future__ import annotations
