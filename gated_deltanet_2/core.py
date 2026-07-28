from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

D_TYPE = jnp.float32


def _recurrent_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    q = q.astype(D_TYPE)
    k = k.astype(D_TYPE)
    v = v.astype(D_TYPE)
    g = g.astype(D_TYPE)
    b = b.astype(D_TYPE)
    w = w.astype(D_TYPE)
    S0 = S0.astype(D_TYPE)

    alpha = jnp.exp(g)
    e = b * k
    z = w * v

    def step(S, inp):
        qt, kt, at, et, zt = inp

        qt = qt[:, None]
        kt = kt[:, None]
        at = at[:, None]
        et = et[:, None]
        zt = zt[:, None]

        S_bar = at * S
        r_t = S_bar.T @ et
        S_new = S_bar + kt * (zt - r_t).T
        o_t = S_new.T @ qt

        return S_new, o_t

    S_final, o = lax.scan(step, S0, (q, k, alpha, e, z))
    return o.squeeze(-1), S_final


def _chunkwise_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    L = k.shape[0]
    C = chunk_size
    if C <= 0 or L % C:
        raise ValueError(
            f"chunk_size={C} must be a positive divisor of the sequence length L={L}"
        )
    N = L // C

    def to_chunks(x):
        return x.reshape(N, C, x.shape[-1]).astype(D_TYPE)

    q = to_chunks(q)
    k = to_chunks(k)
    v = to_chunks(v)
    g = to_chunks(g)
    b = to_chunks(b)
    w = to_chunks(w)
    S0 = S0.astype(D_TYPE)

    G = jnp.cumsum(g, axis=1)
    gamma = jnp.exp(G)
    gamma_C = gamma[:, -1]

    Kbar = k * jnp.exp(-G)
    Ebar = gamma * (b * k)

    Z = w * v
    Qg = gamma * q

    T = jnp.tril(Ebar @ Kbar.swapaxes(-1, -2), k=-1)

    dk = Ebar.shape[-1]
    eye = jnp.eye(T.shape[-1], dtype=T.dtype)

    YU = jax.scipy.linalg.solve_triangular(
        eye + T,
        jnp.concatenate([Ebar, Z], axis=-1),
        lower=True,
        unit_diagonal=True,
    )
    Y, U = YU[..., :dk], YU[..., dk:]

    Aqk = jnp.tril(Qg @ Kbar.swapaxes(-1, -2))
    Ktail = k * (gamma_C[:, None, :] / gamma)

    def chunk_step(S_0, inp):
        Y_n, U_n, Aqk_n, Qg_n, Ktail_n, gamma_C_n = inp

        R = U_n - Y_n @ S_0
        o = Qg_n @ S_0 + Aqk_n @ R
        Sc = gamma_C_n[:, None] * S_0 + Ktail_n.T @ R

        return Sc, o

    S_final, o = lax.scan(chunk_step, S0, (Y, U, Aqk, Qg, Ktail, gamma_C))
    return o.reshape(-1, o.shape[-1]), S_final


def _batchify(fn, **static_kwargs):
    if static_kwargs:
        fn = partial(fn, **static_kwargs)
    over_heads = jax.vmap(fn, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))
    return jax.vmap(over_heads, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))


def recurrent_gated_delta_rule_2(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _batchify(_recurrent_step)(q, k, v, g, b, w, S0)


def chunkwise_gated_delta_rule_2(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    return _batchify(_chunkwise_step)(q, k, v, g, b, w, S0, chunk_size)
