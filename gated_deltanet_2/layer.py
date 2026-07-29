import flax.nnx as nnx
import jax
import jax.numpy as jnp

from gated_deltanet_2.core import (
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)

F32 = jnp.float32

_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, *, eps: float = 1e-5, rngs: nnx.Rngs | None = None):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        xf = x.astype(F32)
        mean = jnp.mean(xf * xf, axis=-1, keepdims=True)
        rms = jax.lax.rsqrt(mean + self.eps)

        return (xf * rms * self.weight[...]).astype(x.dtype)


class GatedRMSNorm(nnx.Module):
    def __init__(
        self,
        head_dim: int,
        d_model: int,
        inner_dim: int,
        gate_rank: int,
        *,
        eps: float = 1e-5,
        rngs: nnx.Rngs,
    ):
        self.norm = RMSNorm(head_dim, eps=eps, rngs=rngs)
        self.gate = nnx.Linear(
            d_model, inner_dim, use_bias=False, kernel_init=_XAVIER, rngs=rngs
        )

    def __call__(self, O_heads: jax.Array, x: jax.Array) -> jax.Array:
        B, L, Hv, dv = O_heads.shape

        o = O_heads.astype(F32)
        o = self.norm(o)

        g = self.gate(x).astype(F32)
        g = jax.nn.sigmoid(g)
        g = g.reshape(B, L, Hv, dv)

        return (o * g).reshape(B, L, Hv * dv)


class ShortConv(nnx.Module):
    def __init__(self, d_model: int, kernel_size: int, *, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(
            in_features=d_model,
            out_features=d_model,
            kernel_size=kernel_size,
            feature_group_count=d_model,
            use_bias=False,
            padding="CAUSAL",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)
