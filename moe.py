import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.nn import initializers


class TopKRouter(nnx.Module):
    def __init__(
        self, d_model: int, num_routed_experts: int, top_k: int = 2, *, rngs: nnx.Rngs
    ):
        self.num_experts = num_routed_experts
        self.top_k = top_k
        # DeepSeek uses weight normalization on the router weights, but
        # standard Linear without bias is functionally close enough.
        self.gate = nnx.Linear(d_model, num_routed_experts, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array):
        logits = self.gate(x)

        # 1. Softmax over ALL routed experts
        probs = jax.nn.softmax(logits, axis=-1)

        # 2. Find Top-K indices
        _, top_k_indices = jax.lax.top_k(logits, k=self.top_k)

        # 3. Gather the weights without renormalizing
        weights = jnp.take_along_axis(probs, top_k_indices, axis=-1)

        # 4. Aux Loss Calculation (Switch Transformer style)
        top1_idx = top_k_indices[:, 0]
        mask_top1 = jax.nn.one_hot(top1_idx, self.num_experts)

        density_f = jnp.mean(mask_top1, axis=0)
        prob_P = jnp.mean(probs, axis=0)
        aux_loss = self.num_experts * jnp.sum(density_f * prob_P)

        return weights, top_k_indices, aux_loss


class VectorizedExperts(nnx.Module):
    """Zero-FLOP Ragged MoE Execution Module for the *routed* experts.

    Identical mechanics to your original — the only conceptual change is
    that `d_ff` here should be the *segmented* (smaller) FFN width, and
    `num_experts` the *segmented* (larger) expert count, per DeepSeekMoE's
    fine-grained segmentation (Sec 3.1 of the paper): split each expert
    into `m` pieces by shrinking d_ff by m and multiplying both
    num_experts and top_k by m, keeping total active params/FLOPs fixed.
    """

    def __init__(self, num_experts: int, d_model: int, d_ff: int, *, rngs: nnx.Rngs):
        self.num_experts = num_experts

        w1_key, w2_key = jax.random.split(rngs.params(), 2)

        self.w1 = nnx.Param(
            initializers.lecun_normal()(w1_key, (num_experts, d_model, d_ff))
        )
        self.b1 = nnx.Param(jnp.zeros((num_experts, d_ff)))

        self.w2 = nnx.Param(
            initializers.lecun_normal()(w2_key, (num_experts, d_ff, d_model))
        )
        self.b2 = nnx.Param(jnp.zeros((num_experts, d_model)))

    def __call__(
        self, x: jax.Array, router_indices: jax.Array, router_weights: jax.Array
    ) -> jax.Array:
        N, _ = x.shape
        top_k = router_indices.shape[1]

        flat_x = jnp.repeat(x, top_k, axis=0)
        flat_indices = router_indices.reshape(-1)
        flat_weights = router_weights.reshape(-1)

        sort_order = jnp.argsort(flat_indices)
        sorted_x = flat_x[sort_order]
        sorted_indices = flat_indices[sort_order]

        group_sizes = jnp.bincount(sorted_indices, length=self.num_experts)

        h = lax.ragged_dot(sorted_x, self.w1.value, group_sizes)
        h = h + self.b1.value[sorted_indices]
        h = nnx.gelu(h)

        expert_outputs = lax.ragged_dot(h, self.w2.value, group_sizes)
        expert_outputs = expert_outputs + self.b2.value[sorted_indices]

        unsort_order = jnp.argsort(sort_order)
        unsorted_outputs = expert_outputs[unsort_order]

        weighted_outputs = unsorted_outputs * flat_weights[:, None]
        weighted_outputs = weighted_outputs.reshape(N, top_k, -1)
        final_output = jnp.sum(weighted_outputs, axis=1)

        return final_output


class SharedExperts(nnx.Module):
    """Dense, always-on shared expert block (DeepSeekMoE Sec 3.2).

    Implemented as K_s parallel FFNs whose outputs are *summed* — in
    practice this is equivalent to (and usually implemented as) a single
    wide dense FFN of width K_s * d_ff_shared, since every token passes
    through all of them unconditionally. No routing, no gating weight,
    no ragged_dot needed — it's a plain dense layer.
    """

    def __init__(
        self,
        d_model: int,
        d_ff_shared: int,
        num_shared_experts: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        # Collapse K_s shared experts of width d_ff_shared into one dense
        # FFN of width K_s * d_ff_shared -- mathematically identical to
        # summing K_s separate FFN outputs, but a single big matmul
        # instead of K_s small ones.
        total_ff = d_ff_shared * num_shared_experts
        self.fc1 = nnx.Linear(d_model, total_ff, rngs=rngs)
        self.fc2 = nnx.Linear(total_ff, d_model, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.gelu(self.fc1(x))
        return self.fc2(h)


class DeepSeekMoE(nnx.Module):
    """DeepSeekMoE layer: fine-grained routed experts + isolated shared experts.

    output = SharedFFN(x) + sum_{i in TopK(x)} g_i * RoutedFFN_i(x)

    Parameters mirror the paper's notation:
      - num_routed_experts (N_r), top_k (K_r): the segmented, sparsely
        routed pool.
      - num_shared_experts (K_s): always-active experts, not gated.
      - d_ff_routed: per-routed-expert hidden width (should already be
        the *segmented*, i.e. shrunk, width -- do the m-way split before
        passing it in).
      - d_ff_shared: per-shared-expert hidden width.
    """

    def __init__(
        self,
        d_model: int,
        d_ff_routed: int,
        d_ff_shared: int,
        num_routed_experts: int,
        num_shared_experts: int = 1,
        top_k: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_routed_experts, top_k=top_k, rngs=rngs)
        self.routed_experts = VectorizedExperts(
            num_routed_experts, d_model, d_ff_routed, rngs=rngs
        )
        self.shared_experts = SharedExperts(
            d_model, d_ff_shared, num_shared_experts, rngs=rngs
        )

    def __call__(self, x: jax.Array):
        # x shape: (batch_size, seq_len, d_model)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])

        # Shared path: dense, unconditional, every token.
        shared_out = self.shared_experts(x_flat)

        # Routed path: sparse, gated, ragged_dot as before.
        weights, indices, aux_loss = self.router(x_flat)
        routed_out = self.routed_experts(x_flat, indices, weights)

        out_flat = shared_out + routed_out

        return out_flat.reshape(orig_shape), aux_loss


if __name__ == "__main__":
    # Sanity check + illustration of the fine-grained segmentation math.
    #
    # Suppose your original (coarse) config was:
    #   num_experts = 8, d_ff = 2048, top_k = 2   (16 experts * ff active per token equiv.)
    #
    # DeepSeekMoE-style, with segmentation factor m=4 and K_s=2 shared experts
    # carved out of the original budget:
    #   d_ff_routed = 2048 // 4       = 512
    #   num_routed_experts = 8 * 4    = 32   (minus whatever you allot to shared)
    #   top_k = 2 * 4                 = 8
    #   d_ff_shared = 512  (matches routed granularity), num_shared_experts = 2

    rngs = nnx.Rngs(0)
    d_model = 256

    layer = DeepSeekMoE(
        d_model=d_model,
        d_ff_routed=512,
        d_ff_shared=512,
        num_routed_experts=32,
        num_shared_experts=2,
        top_k=8,
        rngs=rngs,
    )

    x = jax.random.normal(jax.random.key(1), (2, 16, d_model))
    out, aux = layer(x)
    print("output shape:", out.shape)
    print("aux loss:", aux)
