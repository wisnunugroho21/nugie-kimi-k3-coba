import jax
import jax.numpy as jnp
from flax import nnx

from gated_deltanet_2.core import (
    chunkwise_gated_delta_rule_2,
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


class GatedDeltaNet2(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        num_v_heads: int | None = None,
        chunk_size: int = 64,
        conv_size: int = 4,
        expanded_erase: bool = False,
        compute_dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.compute_dtype = compute_dtype
        self.d_model = d_model
        self.H = num_heads
        self.Hv = num_v_heads or num_heads

        self.group = self.Hv // self.H
        self.dk = head_k_dim
        self.dv = head_v_dim
        self.chunk_size = chunk_size
        self.conv_size = conv_size
        self.expanded_erase = expanded_erase

        qk_proj_dim = self.H * self.dk
        v_proj_dim = self.Hv * self.dv

        self.q_proj = nnx.Linear(
            d_model,
            qk_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            d_model,
            qk_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            d_model,
            v_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.b_proj = nnx.Linear(
            d_model,
            qk_proj_dim,
            use_bias=True,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )  # Proj_b, Eq. 85: b = σ(Proj_b x)
        self.w_proj = nnx.Linear(
            d_model,
            v_proj_dim,
            use_bias=True,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.f_proj = nnx.Linear(
            d_model,
            qk_proj_dim,
            use_bias=True,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        self.q_conv = ShortConv(qk_proj_dim, conv_size, rngs=rngs)
        self.k_conv = ShortConv(qk_proj_dim, conv_size, rngs=rngs)
        self.v_conv = ShortConv(v_proj_dim, conv_size, rngs=rngs)

        self.A_log = nnx.Param(
            jnp.log(jax.random.uniform(rngs.params(), (self.H,), F32, 1.0, 16.0))
        )  # 'a' in -exp(a)·softplus(·)
        dt = jnp.exp(
            jax.random.uniform(
                rngs.params(),
                (self.H * self.dk,),
                F32,
                jnp.log(1e-3),
                jnp.log(1e-1),
            )
        )
        self.dt_bias = nnx.Param(dt + jnp.log(-jnp.expm1(-dt)))

        self.o_norm = GatedRMSNorm(
            head_dim=self.dv,
            d_model=d_model,
            inner_dim=self.Hv * self.dv,
            gate_rank=self.dv,
            rngs=rngs,
        )

        self.o_proj = nnx.Linear(
            v_proj_dim,
            d_model,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

    def _split_qk(self, x: jax.Array, B: int, L: int) -> jax.Array:
        return x.reshape(B, L, self.H, self.dk).swapaxes(1, 2)  # [B,H,L,dk]

    def _split_v(self, x: jax.Array, B: int, L: int) -> jax.Array:
        return x.reshape(B, L, self.H, self.group * self.dv).swapaxes(1, 2)

    def __call__(
        self,
        x: jax.Array,
        return_state: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        B, L, _ = x.shape

        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))

        q, k, v = jax.nn.silu(q), jax.nn.silu(k), jax.nn.silu(v)

        q = self._split_qk(q, B, L)
        k = self._split_qk(k, B, L)
        v = self._split_v(v, B, L)

        # L2-normalize q, k per head (Sec. 3.5 "L2 normalization applied to q_t
        # and k_t"; App. D.2), as x·rsqrt(‖x‖² + ε) — one fused rsqrt instead
        # of a sqrt and a divide; ε guards the all-zero rows SiLU can produce.
        q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)

        # Log-decay branch, computed in fp32 outside the kernel (Eq. 12 / 86; App. C.1 / D.1).
        #   g_t = -exp(a) ⊙ softplus(Proj_f(x_t) + δ),  then α_t = exp(g_t) inside the core.
        f_p = self.f_proj(x).astype(jnp.float32)  # [B,L,H*dk]  Proj_f(x) in Eq. 86
        d_t = self.dt_bias[...].astype(
            jnp.float32
        )  # [H*dk]  per-channel bias δ, Eq. 86
        a_l = self.A_log[...].astype(jnp.float32)  # [H]  'a', per key head (App. C.1)

        f = self._split_qk(f_p + d_t, B, L)  # Proj_f(x)+δ -> [B,H,L,dk]
        a = jnp.exp(a_l)[None, :, None, None]  # exp(a), broadcast over the d_k channels
        g = -a * jax.nn.softplus(f)  # [B,H,L,dk] ≤ 0  (Eq. 86)

        # Channel-wise gates (Eq. 11 / 85).
        b = jax.nn.sigmoid(self.b_proj(x))  # b = σ(Proj_b x) ∈ [0,1]^{d_k}
        b = self._split_qk(b, B, L)

        if self.expanded_erase:
            b = 2.0 * b  # neg-eigenvalue variant: scale ONLY b to [0,2] (Sec. 3.1)

        w = jax.nn.sigmoid(self.w_proj(x))  # w = σ(Proj_w x) ∈ [0,1]^{d_v}
        w = self._split_v(w, B, L)

        S0 = jnp.zeros((B, self.H, self.dk, self.group * self.dv), jnp.float32)

        # Gated Delta Rule-2 chunkwise core (Eq. 10); forms cumsum γ internally (Eq. 30).
        o, S_final = chunkwise_gated_delta_rule_2(
            q,
            k,
            v,
            g,
            b,
            w,
            S0,
            chunk_size=self.chunk_size,
        )

        B, _, L, _ = o.shape
        o = o.swapaxes(1, 2).reshape(B, L, self.Hv, self.dv)  # ungroup value heads
        o = self.o_norm(o, x).astype(
            x.dtype
        )  # low-rank SIGMOID output gate computed inside, from x (see GatedRMSNorm)

        out = self.o_proj(o)  # project back to d_model

        if return_state:
            B = S_final.shape[0]
            return out, (
                S_final.reshape(B, self.H, self.dk, self.group, self.dv)
                .swapaxes(2, 3)  # [B, H, G, dk, dv]
                .reshape(B, self.Hv, self.dk, self.dv)
            )
        return out
