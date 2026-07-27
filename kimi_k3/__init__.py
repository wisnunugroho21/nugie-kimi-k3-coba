"""A readable, small-scale Kimi K3 text backbone built with Flax NNX."""

from kimi_k3.config import KimiK3Config
from kimi_k3.model import (
    AttentionResidual,
    GatedMLA,
    KimiDeltaAttention,
    KimiK3,
    SiTUGLU,
    StableLatentMoE,
    causal_lm_loss,
)

__all__ = [
    "AttentionResidual",
    "GatedMLA",
    "KimiDeltaAttention",
    "KimiK3",
    "KimiK3Config",
    "SiTUGLU",
    "StableLatentMoE",
    "causal_lm_loss",
]
