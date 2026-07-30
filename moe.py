import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.nn import initializers


class TopKRouter(nnx.Module):
    """Top-K Gating router with auxiliary load-balancing loss."""

    def __init__(
        self, d_model: int, num_experts: int, top_k: int = 6, *, rngs: nnx.Rngs
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nnx.Linear(d_model, num_experts, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array):
        logits = self.gate(x)

        # Select top-k fine-grained experts per token
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k=self.top_k)
        weights = jax.nn.softmax(top_k_logits, axis=-1)

        # Auxiliary loss for uniform load balancing across fine-grained experts
        probs = jax.nn.softmax(logits, axis=-1)
        top1_idx = top_k_indices[:, 0]
        mask_top1 = jax.nn.one_hot(top1_idx, self.num_experts)

        density_f = jnp.mean(mask_top1, axis=0)
        prob_P = jnp.mean(probs, axis=0)
        aux_loss = self.num_experts * jnp.sum(density_f * prob_P)

        return weights, top_k_indices, aux_loss


class VectorizedExperts(nnx.Module):
    """Zero-FLOP Ragged MoE Execution Module for Fine-Grained Experts."""

    def __init__(
        self, num_experts: int, d_model: int, d_ff_expert: int, *, rngs: nnx.Rngs
    ):
        self.num_experts = num_experts

        w1_key, w2_key = jax.random.split(rngs.params(), 2)

        # Shape: (E, d_model, d_ff_expert)
        self.w1 = nnx.Param(
            initializers.lecun_normal()(w1_key, (num_experts, d_model, d_ff_expert))
        )
        self.b1 = nnx.Param(jnp.zeros((num_experts, d_ff_expert)))

        # Shape: (E, d_ff_expert, d_model)
        self.w2 = nnx.Param(
            initializers.lecun_normal()(w2_key, (num_experts, d_ff_expert, d_model))
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
        sort_order = jnp.argsort(flat_indices)
        sorted_x = flat_x[sort_order]
        sorted_indices = flat_indices[sort_order]

        # 3. Dynamic group sizes
        group_sizes = jnp.bincount(sorted_indices, length=self.num_experts)

        # 4. COMPUTE: Fine-grained experts execution using ragged_dot
        h = lax.ragged_dot(sorted_x, self.w1.value, group_sizes)
        h = h + self.b1.value[sorted_indices]
        h = nnx.gelu(h)

        expert_outputs = lax.ragged_dot(h, self.w2.value, group_sizes)
        expert_outputs = expert_outputs + self.b2.value[sorted_indices]

        # 5. COMBINE: Unsort tokens back to their original sequential order
        unsort_order = jnp.argsort(sort_order)
        unsorted_outputs = expert_outputs[unsort_order]

        # Apply router weights
        weighted_outputs = unsorted_outputs * flat_weights[:, None]
        weighted_outputs = weighted_outputs.reshape(N, top_k, -1)

        return jnp.sum(weighted_outputs, axis=1)


class SharedExperts(nnx.Module):
    """Unconditionally active shared experts capturing common knowledge."""

    def __init__(
        self, num_shared_experts: int, d_model: int, d_ff_expert: int, *, rngs: nnx.Rngs
    ):
        # Concatenating N_s shared experts into a single projection matrix
        total_shared_dim = num_shared_experts * d_ff_expert
        self.w1 = nnx.Linear(d_model, total_shared_dim, rngs=rngs)
        self.w2 = nnx.Linear(total_shared_dim, d_model, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.gelu(self.w1(x))
        return self.w2(h)


class DeepSeekMoE(nnx.Module):
    """DeepSeekMoE Architecture combining Shared and Fine-Grained Routed Experts."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_routed_experts: int = 64,
        num_shared_experts: int = 2,
        top_k: int = 6,
        split_factor: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        # Fine-grained intermediate dimension per expert
        d_ff_expert = d_ff // split_factor

        # 1. Always active Shared Experts
        self.shared_experts = SharedExperts(
            num_shared_experts=num_shared_experts,
            d_model=d_model,
            d_ff_expert=d_ff_expert,
            rngs=rngs,
        )

        # 2. Sparse Fine-Grained Routed Experts
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_routed_experts,
            top_k=top_k,
            rngs=rngs,
        )
        self.routed_experts = VectorizedExperts(
            num_experts=num_routed_experts,
            d_model=d_model,
            d_ff_expert=d_ff_expert,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])

        # Path 1: Process unconditionally via Shared Experts
        shared_out = self.shared_experts(x_flat)

        # Path 2: Route tokens to fine-grained experts
        weights, indices, aux_loss = self.router(x_flat)
        routed_out = self.routed_experts(x_flat, indices, weights)

        # Combine both expert outputs
        final_out = shared_out + routed_out

        return final_out.reshape(orig_shape), aux_loss
