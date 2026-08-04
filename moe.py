"""DeepSeekMoE (arXiv:2401.06066) in JAX / Flax NNX.

Fine-grained expert segmentation + shared expert isolation, with the
expert-level balance loss from Eq. (14)-(16) of the paper.

Conventions followed:
  * routed experts and shared experts have the SAME intermediate size
    (a shared expert is just a segmented expert that is always active)
  * SwiGLU FFNs, no biases anywhere
  * router runs in fp32, softmax affinities, gates NOT renormalized (V1/V2)
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.nn import initializers

# For stacked expert weights of shape (E, in, out) the expert axis must be
# declared as a batch axis, otherwise it is folded into the receptive field
# and fan_in becomes E * in.
_expert_init = initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,))


class TopKRouter(nnx.Module):
    """Softmax affinity router with the expert-level balance loss."""

    def __init__(
        self,
        d_model: int,
        num_routed_experts: int,
        top_k: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_experts = num_routed_experts
        self.top_k = top_k

        # fp32 router: bf16 softmax over many experts is a known instability.
        self.router = nnx.Linear(
            d_model,
            num_routed_experts,
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, mask: jax.Array | None = None):
        """x: (T, d_model) -> (weights (T, K), indices (T, K), aux_loss scalar).

        `mask`: optional (T,) float/bool array, 1 for real tokens, 0 for padding.
        """
        logits = self.router(x.astype(jnp.float32))
        probs = jax.nn.softmax(logits, axis=-1)  # s_{i,t}

        # softmax is monotonic, so top_k on probs == top_k on logits and we get
        # the gate values for free.
        weights, indices = lax.top_k(probs, k=self.top_k)  # (T, K)

        # ---- expert-level balance loss ------------------------------------
        # f_i = (N'/(K'T)) * sum_t 1[token t selects expert i]   <- ALL K experts
        # P_i = (1/T) * sum_t s_{i,t}
        # L   = sum_i f_i * P_i        (caller multiplies by alpha_1)
        counts = jax.nn.one_hot(indices, self.num_experts, dtype=probs.dtype).sum(
            axis=-2
        )  # (T, E)

        if mask is None:
            f = counts.mean(axis=0)
            P = probs.mean(axis=0)
        else:
            m = mask.astype(probs.dtype)[:, None]
            denom = jnp.maximum(m.sum(), 1.0)
            f = (counts * m).sum(axis=0) / denom
            P = (probs * m).sum(axis=0) / denom

        f = f * (self.num_experts / self.top_k)
        aux_loss = jnp.sum(f * P)

        return weights, indices, aux_loss


class VectorizedExperts(nnx.Module):
    """Dropless grouped-GEMM SwiGLU experts via lax.ragged_dot."""

    def __init__(self, num_experts: int, d_model: int, d_ff: int, *, rngs: nnx.Rngs):
        self.num_experts = num_experts

        k_gate, k_up, k_down = jax.random.split(rngs.params(), 3)
        self.w_gate = nnx.Param(_expert_init(k_gate, (num_experts, d_model, d_ff)))
        self.w_up = nnx.Param(_expert_init(k_up, (num_experts, d_model, d_ff)))
        self.w_down = nnx.Param(_expert_init(k_down, (num_experts, d_ff, d_model)))

    def __call__(
        self, x: jax.Array, router_indices: jax.Array, router_weights: jax.Array
    ) -> jax.Array:
        N, d_model = x.shape
        top_k = router_indices.shape[1]

        flat_indices = router_indices.reshape(-1)  # (N*K,) token-major
        flat_weights = router_weights.reshape(-1)

        # Stable sort -> contiguous per-expert blocks.
        sort_order = jnp.argsort(flat_indices)
        # Flat slot p belongs to token p // top_k, so we can gather straight from
        # x instead of materialising a K-fold repeat of the activations.
        token_ids = sort_order // top_k
        sorted_x = x[token_ids]

        group_sizes = jnp.bincount(flat_indices, length=self.num_experts).astype(
            jnp.int32
        )

        gate = lax.ragged_dot(sorted_x, self.w_gate.value, group_sizes)
        up = lax.ragged_dot(sorted_x, self.w_up.value, group_sizes)
        h = jax.nn.silu(gate) * up
        y = lax.ragged_dot(h, self.w_down.value, group_sizes)

        # Scale by gate values, then scatter-add back to token order.
        y = y * flat_weights[sort_order][:, None].astype(y.dtype)
        out = jnp.zeros((N, d_model), dtype=y.dtype).at[token_ids].add(y)
        return out


class SharedExperts(nnx.Module):
    """K_s always-active experts, fused into one wide SwiGLU FFN.

    Summing K_s independent FFNs is exactly equivalent to one FFN whose
    intermediate dim is K_s * d_ff, provided gate/up are block-concatenated.
    """

    def __init__(
        self,
        d_model: int,
        d_ff_shared: int,
        num_shared_experts: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        total_ff = d_ff_shared * num_shared_experts
        self.w_gate = nnx.Linear(d_model, total_ff, use_bias=False, rngs=rngs)
        self.w_up = nnx.Linear(d_model, total_ff, use_bias=False, rngs=rngs)
        self.w_down = nnx.Linear(total_ff, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w_down(jax.nn.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekMoE(nnx.Module):
    def __init__(
        self,
        d_model: int,
        d_ff_expert: int,
        num_routed_experts: int,
        num_shared_experts: int = 1,
        top_k: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        """`d_ff_expert` is the *segmented* expert width, i.e. d_ff_standard / m.

        Total activated FFN width is (num_shared_experts + top_k) * d_ff_expert,
        which should match the dense baseline you are comparing against.
        """
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_routed_experts, top_k=top_k, rngs=rngs)
        self.routed_experts = VectorizedExperts(
            num_routed_experts, d_model, d_ff_expert, rngs=rngs
        )
        self.shared_experts = (
            SharedExperts(d_model, d_ff_expert, num_shared_experts, rngs=rngs)
            if num_shared_experts > 0
            else None
        )

    def __call__(self, x: jax.Array, mask: jax.Array | None = None):
        """x: (..., d_model). `mask`: (...,) 1 for real tokens, 0 for padding."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        mask_flat = None if mask is None else mask.reshape(-1)

        weights, indices, aux_loss = self.router(x_flat, mask_flat)
        out_flat = self.routed_experts(x_flat, indices, weights).astype(x.dtype)

        if self.shared_experts is not None:
            out_flat = out_flat + self.shared_experts(x_flat)

        return out_flat.reshape(orig_shape), aux_loss
