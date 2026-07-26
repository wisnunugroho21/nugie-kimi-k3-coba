"""A small, readable implementation of LatentMoE in JAX + Flax NNX.

LatentMoE moves only the *routed* experts into a smaller latent space:

    x --router---------------------------------------> top-k expert weights
    |
    +--down projection--> latent routed experts --> up projection--+
    |                                                               +--> output
    +--------------------> full-width shared experts----------------+

This is Equation (2), the paper's recommended accuracy-oriented variant:

    W_up sum_{i in top-k} p_i E_i(W_down x) + sum_j SharedExpert_j(x)

The code implements the architecture and its sparse top-k math, but deliberately
omits production systems machinery such as expert parallelism, all-to-all token
dispatch, capacity limits, quantization, and fused grouped-GEMM kernels.

Paper: https://arxiv.org/abs/2601.18089
Official production implementation:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx


Array = jax.Array


@dataclass(frozen=True)
class LatentMoEConfig:
    """Dimensions for converting a standard MoE into a LatentMoE.

    ``baseline_num_experts`` and ``baseline_top_k`` describe the standard MoE
    that we want to match.  If alpha = d_model / latent_dim, the paper scales:

      * routed expert count: N' = alpha * N, in both variants;
      * active expert count: K' = alpha * K, only in the "accuracy" variant.

    The "accuracy" variant (Equation 2, recommended by the paper) spends the
    savings from latent computation on more active experts.  The "efficiency"
    variant (Equation 1) keeps K unchanged and spends the savings on speed.
    """

    d_model: int
    latent_dim: int
    expert_hidden_dim: int
    baseline_num_experts: int
    baseline_top_k: int
    num_shared_experts: int = 0
    variant: Literal["accuracy", "efficiency"] = "accuracy"

    def __post_init__(self) -> None:
        positive = {
            "d_model": self.d_model,
            "latent_dim": self.latent_dim,
            "expert_hidden_dim": self.expert_hidden_dim,
            "baseline_num_experts": self.baseline_num_experts,
            "baseline_top_k": self.baseline_top_k,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.num_shared_experts < 0:
            raise ValueError("num_shared_experts cannot be negative.")
        if self.d_model % self.latent_dim != 0:
            raise ValueError(
                "This educational implementation requires d_model / latent_dim "
                "to be an integer compression ratio."
            )
        if self.baseline_top_k > self.baseline_num_experts:
            raise ValueError("baseline_top_k cannot exceed baseline_num_experts.")
        if self.variant not in ("accuracy", "efficiency"):
            raise ValueError("variant must be 'accuracy' or 'efficiency'.")

    @property
    def compression_ratio(self) -> int:
        """alpha = d / ell in the paper."""
        return self.d_model // self.latent_dim

    @property
    def num_routed_experts(self) -> int:
        """N' = alpha * N: more, narrower experts at similar parameter cost."""
        return self.compression_ratio * self.baseline_num_experts

    @property
    def top_k(self) -> int:
        """K' = alpha * K for accuracy, or K for efficiency."""
        multiplier = self.compression_ratio if self.variant == "accuracy" else 1
        return multiplier * self.baseline_top_k


class RoutingInfo(NamedTuple):
    """Values useful for inspecting routing or adding a training loss."""

    probabilities: Array  # [..., N']: softmax probabilities for every expert
    expert_indices: Array  # [..., K']: selected expert IDs
    expert_weights: Array  # [..., K']: selected (not renormalized) probabilities
    load_balancing_loss: Array  # scalar; multiply by a small coefficient in training


def _xavier_uniform(key: Array, shape: tuple[int, int, int]) -> Array:
    """Xavier initialization for a bank of [expert, input, output] matrices."""
    _, fan_in, fan_out = shape
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


class ExpertBank(nnx.Module):
    """A collection of SwiGLU experts stored in three stacked parameter arrays.

    One expert computes:

        E(x) = W_down (SiLU(W_gate x) * (W_up x))

    Stacking the expert dimension makes it easy to gather just the top-k expert
    matrices for each token.  That keeps the mathematical implementation sparse:
    we evaluate K selected experts, rather than all N experts.
    """

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_experts = num_experts
        self.input_dim = input_dim

        # [E, D, M], [E, D, M], and [E, M, D] correspond to the paper's
        # W_gate, W_FC1, and W_FC2 (transposed to suit JAX's row-vector math).
        self.gate_kernel = nnx.Param(
            _xavier_uniform(rngs.params(), (num_experts, input_dim, hidden_dim))
        )
        self.up_kernel = nnx.Param(
            _xavier_uniform(rngs.params(), (num_experts, input_dim, hidden_dim))
        )
        self.down_kernel = nnx.Param(
            _xavier_uniform(rngs.params(), (num_experts, hidden_dim, input_dim))
        )

    def selected(self, x: Array, expert_indices: Array) -> Array:
        """Run only the selected experts.

        Args:
            x: [T, D] flattened tokens.
            expert_indices: [T, K] expert IDs selected independently per token.

        Returns:
            [T, K, D], one output per token/expert pair.
        """
        # Advanced indexing gathers a small per-token bank:
        # [E, D, M] -> [T, K, D, M].
        gate_kernel = self.gate_kernel[...][expert_indices]
        up_kernel = self.up_kernel[...][expert_indices]
        down_kernel = self.down_kernel[...][expert_indices]

        gate = jnp.einsum("td,tkdm->tkm", x, gate_kernel)
        value = jnp.einsum("td,tkdm->tkm", x, up_kernel)
        hidden = jax.nn.silu(gate) * value
        return jnp.einsum("tkm,tkmd->tkd", hidden, down_kernel)

    def all(self, x: Array) -> Array:
        """Run every expert in this bank and sum them.

        This path is used only for the small number S of *shared* experts.  They
        always see every token, exactly as the second term of Equations (1)-(2).
        """
        gate = jnp.einsum("td,edm->tem", x, self.gate_kernel[...])
        value = jnp.einsum("td,edm->tem", x, self.up_kernel[...])
        hidden = jax.nn.silu(gate) * value
        per_expert = jnp.einsum("tem,emd->ted", hidden, self.down_kernel[...])
        return per_expert.sum(axis=1)


class LatentMoE(nnx.Module):
    """LatentMoE feed-forward layer from Equations (1) and (2).

    Important details preserved from the paper and NVIDIA's implementation:

      1. The router reads the original d-dimensional token, not its latent.
      2. Only the routed path is projected down to ell dimensions.
      3. Routing, expert computation, and aggregation precede W_up.
      4. Shared experts operate on the original d-dimensional token and are
         added only after the routed result is projected back to d.

    This module returns the MoE branch itself.  A Transformer block would
    normally apply normalization before it and add a residual connection after it.
    """

    def __init__(self, config: LatentMoEConfig, *, rngs: nnx.Rngs):
        self.config = config

        # W_r' in the paper: routing remains full-width because its cost is small
        # relative to expert weight loading and token dispatch.
        self.router = nnx.Linear(
            config.d_model,
            config.num_routed_experts,
            use_bias=False,
            rngs=rngs,
        )

        # Shared W_down and W_up surround the entire routed-expert mixture.
        self.down_projection = nnx.Linear(
            config.d_model,
            config.latent_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.up_projection = nnx.Linear(
            config.latent_dim,
            config.d_model,
            use_bias=False,
            rngs=rngs,
        )

        # Routed experts are narrow: their input/output dimension is ell, while
        # their nonlinear intermediate width M stays unchanged.
        self.routed_experts = ExpertBank(
            config.num_routed_experts,
            config.latent_dim,
            config.expert_hidden_dim,
            rngs=rngs,
        )

        # Shared experts intentionally remain full-width (dimension d).
        self.shared_experts = (
            ExpertBank(
                config.num_shared_experts,
                config.d_model,
                config.expert_hidden_dim,
                rngs=rngs,
            )
            if config.num_shared_experts > 0
            else None
        )

    def __call__(self, x: Array) -> tuple[Array, RoutingInfo]:
        """Apply LatentMoE to tokens shaped ``[..., d_model]``."""
        if x.ndim < 2 or x.shape[-1] != self.config.d_model:
            raise ValueError(
                f"x must have shape [..., {self.config.d_model}], got {x.shape}."
            )

        leading_shape = x.shape[:-1]
        if math.prod(leading_shape) == 0:
            raise ValueError("LatentMoE requires at least one input token.")
        tokens = x.reshape(-1, self.config.d_model)  # [T, d]

        # p' = Softmax(W_r' x), computed from the ORIGINAL token (paper Sec. 3).
        # Float32 softmax avoids losing small routing probabilities under bf16.
        router_logits = self.router(tokens).astype(jnp.float32)
        probabilities = jax.nn.softmax(router_logits, axis=-1)  # [T, N']

        # T_{K',N'}: choose K' experts.  The selected values are the original
        # softmax probabilities from all N' experts, as written in Equation (2);
        # they are deliberately not renormalized after top-k.
        expert_weights, expert_indices = jax.lax.top_k(
            probabilities, self.config.top_k
        )  # both [T, K']

        # W_down x enters the latent space before any token/expert computation.
        latent_tokens = self.down_projection(tokens)  # [T, ell]

        # E_i(W_down x), evaluated for only the selected token/expert pairs.
        selected_outputs = self.routed_experts.selected(
            latent_tokens, expert_indices
        )  # [T, K', ell]

        # Sum_i p_i E_i(...), still entirely in latent space.
        mixed_latents = jnp.einsum(
            "tk,tkd->td", expert_weights, selected_outputs
        )  # [T, ell]

        # W_up is applied after expert aggregation, matching both the equation
        # and the production order (combine in latent space, then project up).
        output = self.up_projection(mixed_latents)  # [T, d]

        # Full-width shared experts form a separate dense path.
        if self.shared_experts is not None:
            output = output + self.shared_experts.all(tokens)

        # A simple differentiable load-balancing auxiliary loss.  It is not part
        # of the forward equation, but the paper's training runs use balancing.
        # It equals 1 at perfectly uniform routing; callers may add, for example,
        # 1e-4 * load_balancing_loss to the task loss.
        assignment = jax.nn.one_hot(
            expert_indices, self.config.num_routed_experts
        )  # [T, K', N']
        assignment_fraction = assignment.mean(axis=(0, 1))  # sums to 1
        mean_probability = probabilities.mean(axis=0)  # sums to 1
        load_balancing_loss = self.config.num_routed_experts * jnp.sum(
            assignment_fraction * mean_probability
        )

        routing = RoutingInfo(
            probabilities=probabilities.reshape(
                *leading_shape, self.config.num_routed_experts
            ),
            expert_indices=expert_indices.reshape(
                *leading_shape, self.config.top_k
            ),
            expert_weights=expert_weights.reshape(
                *leading_shape, self.config.top_k
            ),
            load_balancing_loss=load_balancing_loss,
        )
        return output.reshape(*leading_shape, self.config.d_model), routing


if __name__ == "__main__":
    # Tiny demonstration.  A 4x compression turns the baseline's N=4, K=1 into
    # N'=16, K'=4 for the recommended accuracy variant.
    config = LatentMoEConfig(
        d_model=64,
        latent_dim=16,
        expert_hidden_dim=32,
        baseline_num_experts=4,
        baseline_top_k=1,
        num_shared_experts=1,
        variant="accuracy",
    )
    model = LatentMoE(config, rngs=nnx.Rngs(0))

    x = jax.random.normal(jax.random.key(1), (2, 8, config.d_model))
    y, routing = model(x)

    print("compression ratio:", config.compression_ratio)
    print("routed experts / active experts:", config.num_routed_experts, config.top_k)
    print("input / output shape:", x.shape, y.shape)
    print("first token's selected experts:", routing.expert_indices[0, 0])
    print("load-balancing loss:", float(routing.load_balancing_loss))
