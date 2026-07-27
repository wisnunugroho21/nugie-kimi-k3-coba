"""Configuration for the educational Kimi K3 implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AttentionKind = Literal["kda", "mla"]


@dataclass(frozen=True)
class KimiK3Config:
    """Describe either a runnable miniature or the paper-scale Kimi K3.

    The defaults intentionally produce a very small model. Calling
    :meth:`paper` returns the dimensions from Table 1 and the official released
    config, but constructing that model is far beyond a normal workstation.

    Layer numbering in the paper and official config starts at one. Each
    four-layer hybrid block is ``KDA, KDA, KDA, MLA`` and the last layer is
    always MLA, even when it follows a complete hybrid block.
    """

    vocab_size: int = 256
    hidden_size: int = 64
    num_layers: int = 5

    num_heads: int = 4
    head_dim: int = 16
    q_lora_rank: int = 32
    kv_lora_rank: int = 32
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0

    dense_hidden_size: int = 128
    latent_moe_dim: int = 32
    moe_hidden_size: int = 64
    num_experts: int = 8
    num_experts_per_token: int = 2
    num_shared_experts: int = 2
    first_dense_layers: int = 1

    attn_res_block_size: int = 4
    situ_gate_beta: float = 4.0
    situ_up_beta: float = 25.0
    rms_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    max_sequence_length: int = 1024

    def __post_init__(self) -> None:
        """Reject shapes that would make the equations ambiguous or invalid."""
        positive_fields = (
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_heads",
            "head_dim",
            "q_lora_rank",
            "kv_lora_rank",
            "short_conv_kernel_size",
            "dense_hidden_size",
            "latent_moe_dim",
            "moe_hidden_size",
            "num_experts",
            "num_experts_per_token",
            "num_shared_experts",
            "attn_res_block_size",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        if self.num_experts_per_token >= self.num_experts:
            raise ValueError(
                "num_experts_per_token must be smaller than num_experts; "
                "Quantile Balancing needs the (k + 1)-th score"
            )
        if not 0 <= self.first_dense_layers <= self.num_layers:
            raise ValueError("first_dense_layers must be between 0 and num_layers")
        if self.gate_lower_bound >= 0:
            raise ValueError("gate_lower_bound must be negative")

    def attention_kind(self, zero_based_layer: int) -> AttentionKind:
        """Return ``"kda"`` or ``"mla"`` for one decoder layer.

        This exactly captures the released K3 schedule: global MLA occurs every
        fourth layer and once more at the end of the backbone.
        """
        if not 0 <= zero_based_layer < self.num_layers:
            raise IndexError(f"layer index {zero_based_layer} is out of range")
        layer_number = zero_based_layer + 1
        if layer_number % 4 == 0 or layer_number == self.num_layers:
            return "mla"
        return "kda"

    @classmethod
    def tiny(cls, **overrides: object) -> "KimiK3Config":
        """Return the runnable default config with selected fields replaced."""
        values = cls().__dict__ | overrides
        return cls(**values)

    @classmethod
    def paper(cls, **overrides: object) -> "KimiK3Config":
        """Return the 2.78T model's published text-backbone dimensions.

        This helper is documentary. Initializing it locally would allocate
        trillions of parameters; use :meth:`tiny` for examples and tests.
        """
        values: dict[str, object] = {
            "vocab_size": 163_840,
            "hidden_size": 7_168,
            "num_layers": 93,
            "num_heads": 96,
            "head_dim": 128,
            "q_lora_rank": 1_536,
            "kv_lora_rank": 512,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "dense_hidden_size": 33_792,
            "latent_moe_dim": 3_584,
            "moe_hidden_size": 3_072,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "num_shared_experts": 2,
            "first_dense_layers": 1,
            "attn_res_block_size": 12,
            "situ_gate_beta": 4.0,
            "situ_up_beta": 25.0,
            "rms_norm_eps": 1e-5,
            "initializer_std": 0.02,
            "max_sequence_length": 1_048_576,
        }
        values.update(overrides)
        return cls(**values)
