"""A small, readable JAX/Flax NNX implementation of Gated DeltaNet-2.

The two functions below implement Gated Delta Rule-2 from
"Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention":

    gated_delta_rule_2_recurrent  -- Eq. (9), one token at a time.
    gated_delta_rule_2_chunkwise  -- Eqs. (18)-(25), using the WY form.

This is reference code, not a replacement for the official fused Triton
kernels.  It favors a direct correspondence with the paper over speed.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from flax import nnx


Array = jax.Array


def _initial_state(q: Array, v: Array, state: Array | None) -> Array:
    """Return an fp32 state with shape [batch, heads, key_dim, value_dim]."""
    if state is not None:
        return state.astype(jnp.float32)
    return jnp.zeros(
        (q.shape[0], q.shape[2], q.shape[3], v.shape[3]),
        dtype=jnp.float32,
    )


def _check_shapes(
    q: Array,
    k: Array,
    v: Array,
    log_decay: Array,
    erase_gate: Array,
    write_gate: Array,
) -> None:
    """Catch the most common axis mix-ups before an einsum does."""
    if q.shape != k.shape or q.shape != log_decay.shape or q.shape != erase_gate.shape:
        raise ValueError("q, k, log_decay, and erase_gate must have the same [B,T,H,K] shape")
    if v.shape != write_gate.shape:
        raise ValueError("v and write_gate must have the same [B,T,H,V] shape")
    if q.shape[:3] != v.shape[:3]:
        raise ValueError("q/k and v must share their batch, time, and head axes")


def gated_delta_rule_2_recurrent(
    q: Array,
    k: Array,
    v: Array,
    log_decay: Array,
    erase_gate: Array,
    write_gate: Array,
    initial_state: Array | None = None,
) -> tuple[Array, Array]:
    """Apply Gated Delta Rule-2 token by token.

    All inputs are already projected and have shapes:

        q, k, log_decay, erase_gate: [batch, time, heads, key_dim]
        v, write_gate:               [batch, time, heads, value_dim]

    ``log_decay`` is g_t from the paper, so alpha_t = exp(g_t).  It should
    normally be non-positive.  The returned state has shape [B,H,K,V].

    This is Eq. (9) written as three easy-to-read operations:

        S_bar = Diag(alpha_t) S_{t-1}
        residual = (w_t * v_t) - S_bar.T (b_t * k_t)
        S_t = S_bar + k_t residual.T
    """
    _check_shapes(q, k, v, log_decay, erase_gate, write_gate)
    output_dtype = v.dtype
    state = _initial_state(q, v, initial_state)

    # lax.scan iterates over its leading axis, so move time to the front.
    tokens = tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, log_decay, erase_gate, write_gate))

    def step(state: Array, token: tuple[Array, ...]) -> tuple[Array, Array]:
        q_t, k_t, v_t, g_t, b_t, w_t = token

        # D_t S_{t-1}: decay is channel-wise along the key (row) axis.
        decayed_state = jnp.exp(g_t)[..., :, None] * state

        # e_t = b_t * k_t chooses which old key channels should be erased;
        # z_t = w_t * v_t independently chooses which values are written.
        e_t = b_t * k_t
        z_t = w_t * v_t
        old_value = jnp.einsum("bhk,bhkv->bhv", e_t, decayed_state)
        residual = z_t - old_value

        # A rank-one update writes the residual in direction k_t.
        state = decayed_state + jnp.einsum("bhk,bhv->bhkv", k_t, residual)

        # The paper reads after updating the state: o_t = S_t^T q_t.
        output = jnp.einsum("bhk,bhkv->bhv", q_t, state)
        return state, output

    final_state, output = jax.lax.scan(step, state, tokens)
    return jnp.moveaxis(output, 0, 1).astype(output_dtype), final_state


def gated_delta_rule_2_chunkwise(
    q: Array,
    k: Array,
    v: Array,
    log_decay: Array,
    erase_gate: Array,
    write_gate: Array,
    initial_state: Array | None = None,
    *,
    chunk_size: int = 64,
) -> tuple[Array, Array]:
    """Apply the paper's chunkwise WY form of Gated Delta Rule-2.

    The public shapes and result are identical to the recurrent function.
    Tokens inside a chunk are evaluated with matrix products and one small
    triangular solve; only the state passed between chunks is recurrent.

    For clarity, this implementation pads the last chunk with no-op tokens.
    ``chunk_size`` is expected to be a small static integer (64 in the paper).
    The decay normalization uses fp32, as does the official implementation.
    """
    _check_shapes(q, k, v, log_decay, erase_gate, write_gate)
    output_dtype = v.dtype
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    state = _initial_state(q, v, initial_state)
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    pad = (-length) % chunk_size

    # A padded token must leave the state unchanged: g=0 gives decay 1, and
    # zero q/k/v/gates gives no erase, write, or output.
    def pad_time(x: Array) -> Array:
        return jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))

    padded = tuple(pad_time(x) for x in (q, k, v, log_decay, erase_gate, write_gate))
    num_chunks = (length + pad) // chunk_size

    # Reshape [B, T, H, D] -> [num_chunks, B, H, C, D] for lax.scan.
    def split_chunks(x: Array) -> Array:
        x = x.reshape(batch, num_chunks, chunk_size, heads, x.shape[-1])
        return x.transpose(1, 0, 3, 2, 4)

    chunks = tuple(split_chunks(x) for x in padded)

    def chunk_step(state: Array, chunk: tuple[Array, ...]) -> tuple[Array, Array]:
        q_c, k_c, v_c, g_c, b_c, w_c = chunk  # each is [B,H,C,D]

        # gamma_r = exp(sum_{i<=r} g_i).  Dividing the state by gamma turns
        # the decayed recurrence into a plain asymmetric delta recurrence.
        gamma = jnp.exp(jnp.cumsum(g_c.astype(jnp.float32), axis=2))
        k_bar = k_c / gamma
        e_bar = gamma * (b_c * k_c)
        z = w_c * v_c

        # T_rs = e_bar_r^T k_bar_s for s < r.  It is strictly lower
        # triangular, so I+T has a unit diagonal and is cheap to solve.
        interaction = jnp.einsum("bhck,bhsk->bhcs", e_bar, k_bar)
        interaction = jnp.tril(interaction, k=-1)
        eye = jnp.eye(chunk_size, dtype=interaction.dtype)
        triangular = interaction + eye[None, None, :, :]

        # A=(I+T)^-1 is never formed explicitly.  These two solves compute
        # Y=A e_bar and U=A z, exactly as in Eq. (22).
        y = jsp.linalg.solve_triangular(
            triangular, e_bar, lower=True, unit_diagonal=True
        )
        u = jsp.linalg.solve_triangular(
            triangular, z, lower=True, unit_diagonal=True
        )
        residual = u - jnp.einsum("bhck,bhkv->bhcv", y, state)

        # Eq. (24): all outputs in this chunk are computed in parallel.
        q_gamma = gamma * q_c
        causal_scores = jnp.einsum("bhck,bhsk->bhcs", q_gamma, k_bar)
        causal_scores = jnp.tril(causal_scores)  # include the current token
        output = (
            jnp.einsum("bhck,bhkv->bhcv", q_gamma, state)
            + jnp.einsum("bhcs,bhsv->bhcv", causal_scores, residual)
        )

        # Eq. (23): advance the recurrent state once per whole chunk.
        gamma_last = gamma[:, :, -1, :]
        k_tail = gamma_last[:, :, None, :] * k_bar
        state = (
            gamma_last[..., :, None] * state
            + jnp.einsum("bhck,bhcv->bhkv", k_tail, residual)
        )
        return state, output

    final_state, output_chunks = jax.lax.scan(chunk_step, state, chunks)

    # [chunks,B,H,C,V] -> [B,T,H,V], then discard the padded outputs.
    output = output_chunks.transpose(1, 0, 3, 2, 4)
    output = output.reshape(batch, num_chunks * chunk_size, heads, value_dim)
    return output[:, :length].astype(output_dtype), final_state


class GatedDeltaNet2(nnx.Module):
    """Minimal learnable Gated DeltaNet-2 token mixer built with Flax NNX.

    The official layer also uses short depth-wise convolutions and fused GPU
    kernels.  They are deliberately omitted here: the projections and update
    rule are retained, while every important tensor remains visible.

    Args:
        model_dim: Width of input and output token embeddings.
        num_heads: Number of recurrent memory heads.
        key_dim: Per-head key/query width.
        value_dim: Per-head value width (defaults to ``key_dim``).
        rngs: Flax NNX random streams, e.g. ``nnx.Rngs(0)``.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        key_dim: int,
        value_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        value_dim = key_dim if value_dim is None else value_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

        key_width = num_heads * key_dim
        value_width = num_heads * value_dim

        # The official code initializes Linear weights with Xavier uniform and
        # gain 2^-2.5.  variance_scaling receives the squared gain.
        kernel_init = jax.nn.initializers.variance_scaling(
            scale=2.0**-5, mode="fan_avg", distribution="uniform"
        )
        linear = lambda din, dout, bias=False: nnx.Linear(  # noqa: E731
            din,
            dout,
            use_bias=bias,
            kernel_init=kernel_init,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

        self.q_proj = linear(model_dim, key_width)
        self.k_proj = linear(model_dim, key_width)
        self.v_proj = linear(model_dim, value_width)

        # Independent channel-wise erase and write gates are the defining
        # difference from KDA's one tied scalar beta.
        self.erase_proj = linear(model_dim, key_width)
        self.write_proj = linear(model_dim, value_width)

        # Match the official small two-layer projections for decay and the
        # output gate.  value_dim is the inexpensive bottleneck width.
        self.decay_in = linear(model_dim, value_dim)
        self.decay_out = linear(value_dim, key_width)
        self.output_gate_in = linear(model_dim, value_dim)
        self.output_gate_out = linear(value_dim, value_width, bias=True)

        # g_t = -exp(a) * softplus(f(x_t) + delta).  exp(g_t) is therefore
        # in (0,1], giving every key channel its own stable decay factor.
        a_key, dt_key = jax.random.split(rngs.params())
        a = jax.random.uniform(a_key, (num_heads,), minval=1.0, maxval=16.0)
        dt = jnp.exp(
            jax.random.uniform(
                dt_key,
                (num_heads, key_dim),
                minval=jnp.log(0.001),
                maxval=jnp.log(0.1),
            )
        )
        inverse_softplus_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.a_log = nnx.Param(jnp.log(a))
        self.dt_bias = nnx.Param(inverse_softplus_dt)

        self.output_norm = nnx.RMSNorm(value_dim, epsilon=1e-5, rngs=rngs)
        self.output_proj = linear(value_width, model_dim)

    def __call__(
        self,
        x: Array,
        *,
        mode: Literal["recurrent", "chunkwise"] = "chunkwise",
        initial_state: Array | None = None,
        chunk_size: int = 64,
    ) -> tuple[Array, Array]:
        """Mix ``x[B,T,model_dim]`` and return ``(output, final_state)``."""
        if x.ndim != 3 or x.shape[-1] != self.model_dim:
            raise ValueError(f"x must have shape [B,T,{self.model_dim}]")
        batch, length, _ = x.shape

        def key_heads(y: Array) -> Array:
            return y.reshape(batch, length, self.num_heads, self.key_dim)

        def value_heads(y: Array) -> Array:
            return y.reshape(batch, length, self.num_heads, self.value_dim)

        # The official layer applies SiLU after q/k/v projections (or its
        # short convolution), then L2-normalizes q and k per head.
        q = key_heads(jax.nn.silu(self.q_proj(x)))
        k = key_heads(jax.nn.silu(self.k_proj(x)))
        v = value_heads(jax.nn.silu(self.v_proj(x)))
        q = q / jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k / jnp.sqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q = q * self.key_dim**-0.5

        erase_gate = jax.nn.sigmoid(key_heads(self.erase_proj(x)))
        write_gate = jax.nn.sigmoid(value_heads(self.write_proj(x)))

        decay_logits = key_heads(self.decay_out(self.decay_in(x)))
        rate = jnp.exp(self.a_log.value)[None, None, :, None]
        log_decay = -rate * jax.nn.softplus(
            decay_logits.astype(jnp.float32) + self.dt_bias.value[None, None]
        )

        if mode == "recurrent":
            output, final_state = gated_delta_rule_2_recurrent(
                q, k, v, log_decay, erase_gate, write_gate, initial_state
            )
        elif mode == "chunkwise":
            output, final_state = gated_delta_rule_2_chunkwise(
                q,
                k,
                v,
                log_decay,
                erase_gate,
                write_gate,
                initial_state,
                chunk_size=chunk_size,
            )
        else:
            raise ValueError("mode must be 'recurrent' or 'chunkwise'")

        # As in the official layer: gated RMS normalization, then project the
        # heads back to model_dim.  RMSNorm acts on each head's last axis.
        output_gate = value_heads(self.output_gate_out(self.output_gate_in(x)))
        output = self.output_norm(output) * jax.nn.silu(output_gate)
        output = output.reshape(batch, length, self.num_heads * self.value_dim)
        return self.output_proj(output), final_state


__all__ = [
    "GatedDeltaNet2",
    "gated_delta_rule_2_chunkwise",
    "gated_delta_rule_2_recurrent",
]
