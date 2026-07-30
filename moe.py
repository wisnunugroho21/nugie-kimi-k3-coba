import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.nn import initializers


class TopKRouter(nnx.Module):
    """Top-K Gating router with auxiliary load-balancing loss."""

    def __init__(
        self, d_model: int, num_experts: int, top_k: int = 2, *, rngs: nnx.Rngs
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nnx.Linear(d_model, num_experts, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array):
        logits = self.gate(x)

        # Select top-k experts per token
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k=self.top_k)
        weights = jax.nn.softmax(top_k_logits, axis=-1)

        # Auxiliary loss for uniform load balancing
        probs = jax.nn.softmax(logits, axis=-1)
        top1_idx = top_k_indices[:, 0]
        mask_top1 = jax.nn.one_hot(top1_idx, self.num_experts)

        density_f = jnp.mean(mask_top1, axis=0)
        prob_P = jnp.mean(probs, axis=0)
        aux_loss = self.num_experts * jnp.sum(density_f * prob_P)

        return weights, top_k_indices, aux_loss


class VectorizedExperts(nnx.Module):
    """Zero-FLOP Ragged MoE Execution Module."""

    def __init__(self, num_experts: int, d_model: int, d_ff: int, *, rngs: nnx.Rngs):
        self.num_experts = num_experts

        w1_key, w2_key = jax.random.split(rngs.params(), 2)

        # Shape: (E, d_model, d_ff)
        self.w1 = nnx.Param(
            initializers.lecun_normal()(w1_key, (num_experts, d_model, d_ff))
        )
        self.b1 = nnx.Param(jnp.zeros((num_experts, d_ff)))

        # Shape: (E, d_ff, d_model)
        self.w2 = nnx.Param(
            initializers.lecun_normal()(w2_key, (num_experts, d_ff, d_model))
        )
        self.b2 = nnx.Param(jnp.zeros((num_experts, d_model)))

    def __call__(
        self, x: jax.Array, router_indices: jax.Array, router_weights: jax.Array
    ) -> jax.Array:
        N, _ = x.shape
        top_k = router_indices.shape[1]

        # 1. Flatten tokens for independent top-k routing
        flat_x = jnp.repeat(x, top_k, axis=0)
        flat_indices = router_indices.reshape(-1)
        flat_weights = router_weights.reshape(-1)

        # 2. SORT: Group tokens contiguously by assigned expert
        # This is a requirement for ragged_dot
        sort_order = jnp.argsort(flat_indices)
        sorted_x = flat_x[sort_order]
        sorted_indices = flat_indices[sort_order]

        # 3. Calculate dynamic group sizes (token count per expert)
        # Setting 'length' explicitly ensures the output shape (num_experts,) is static
        # and compatible with JAX JIT compilation.
        group_sizes = jnp.bincount(sorted_indices, length=self.num_experts)

        # 4. COMPUTE: True Zero-FLOP Batched FFN using ragged_dot
        # First FFN Layer:
        # (Total_Tokens, d_model) @ (E, d_model, d_ff) -> (Total_Tokens, d_ff)
        h = lax.ragged_dot(sorted_x, self.w1.value, group_sizes)

        # Add bias (gather the correct bias for each token's chosen expert)
        h = h + self.b1.value[sorted_indices]
        h = nnx.gelu(h)

        # Second FFN Layer:
        # (Total_Tokens, d_ff) @ (E, d_ff, d_model) -> (Total_Tokens, d_model)
        expert_outputs = lax.ragged_dot(h, self.w2.value, group_sizes)
        expert_outputs = expert_outputs + self.b2.value[sorted_indices]

        # 5. COMBINE: Unsort tokens back to their original sequential order
        unsort_order = jnp.argsort(sort_order)
        unsorted_outputs = expert_outputs[unsort_order]

        # Apply the router weights
        weighted_outputs = unsorted_outputs * flat_weights[:, None]

        # Reshape to (N, top_k, d_model) and sum the top-k components
        weighted_outputs = weighted_outputs.reshape(N, top_k, -1)
        final_output = jnp.sum(weighted_outputs, axis=1)

        return final_output


class RoutedMoE(nnx.Module):
    """Lossless, Dropless Mixture of Experts Layer."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k=top_k, rngs=rngs)
        self.experts = VectorizedExperts(num_experts, d_model, d_ff, rngs=rngs)

    def __call__(self, x: jax.Array):
        # x shape: (batch_size, seq_len, d_model)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])

        # Route tokens
        weights, indices, aux_loss = self.router(x_flat)

        # Sparse Evaluation using True Zero-FLOP Ragged Dot
        out_flat = self.experts(x_flat, indices, weights)

        # Reshape back to original sequence length
        return out_flat.reshape(orig_shape), aux_loss
