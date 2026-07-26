"""Numerical checks for the Gated Delta Rule-2 recurrence cores."""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from gated_deltanet_2.core import (
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)


def _inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    B, H, L, dk, dv = 1, 2, 8, 3, 4
    q = rng.normal(size=(B, H, L, dk))
    k = rng.normal(size=(B, H, L, dk))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    arrays = (
        q,
        k,
        rng.normal(size=(B, H, L, dv)),
        -rng.uniform(0.0, 0.2, size=(B, H, L, dk)),
        rng.uniform(0.0, 1.0, size=(B, H, L, dk)),
        rng.uniform(0.0, 1.0, size=(B, H, L, dv)),
        rng.normal(size=(B, H, dk, dv)),
    )
    return tuple(jnp.asarray(x, dtype=jnp.float32) for x in arrays)


def _float64_oracle(*args):
    """Independent NumPy implementation of paper Eqs. 8-10."""
    q, k, v, g, b, w, S0 = (np.asarray(x, dtype=np.float64) for x in args)
    B, H, L, _ = q.shape
    out = np.empty((*q.shape[:3], v.shape[-1]), dtype=np.float64)
    final = np.empty_like(S0)

    for batch in range(B):
        for head in range(H):
            S = S0[batch, head].copy()
            for token in range(L):
                S_bar = np.exp(g[batch, head, token])[:, None] * S
                e = b[batch, head, token] * k[batch, head, token]
                z = w[batch, head, token] * v[batch, head, token]
                S = S_bar + np.outer(
                    k[batch, head, token],
                    z - S_bar.T @ e,
                )
                out[batch, head, token] = S.T @ q[batch, head, token]
            final[batch, head] = S
    return out, final


@pytest.mark.parametrize(
    ("core", "sub_chunk_size"),
    [
        ("faithful", 2),
        ("stacked_rhs", 2),
        ("centered", 2),
        ("pairwise", 2),
        ("subchunking", 2),
    ],
)
def test_chunkwise_cores_match_float64_oracle(core, sub_chunk_size):
    args = _inputs()
    expected_o, expected_s = _float64_oracle(*args)
    actual_o, actual_s = chunkwise_gated_delta_rule_2(
        *args,
        chunk_size=4,
        core=core,
        sub_chunk_size=sub_chunk_size,
    )
    np.testing.assert_allclose(actual_o, expected_o, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual_s, expected_s, rtol=2e-5, atol=2e-6)


def test_recurrent_core_matches_float64_oracle():
    args = _inputs(seed=1)
    expected_o, expected_s = _float64_oracle(*args)
    actual_o, actual_s = recurrent_gated_delta_rule_2(*args)
    np.testing.assert_allclose(actual_o, expected_o, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_s, expected_s, rtol=1e-5, atol=1e-6)


def test_centered_gradients_match_recurrent_reference():
    args = _inputs(seed=2)

    def recurrent_loss(*xs):
        o, state = recurrent_gated_delta_rule_2(*xs)
        return jnp.sum(o * o) + jnp.sum(state * state)

    def chunkwise_loss(*xs):
        o, state = chunkwise_gated_delta_rule_2(
            *xs,
            chunk_size=4,
            core="centered",
        )
        return jnp.sum(o * o) + jnp.sum(state * state)

    argnums = tuple(range(len(args)))
    expected = jax.grad(recurrent_loss, argnums=argnums)(*args)
    actual = jax.grad(chunkwise_loss, argnums=argnums)(*args)
    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            actual_grad,
            expected_grad,
            rtol=3e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("core", ["pairwise", "subchunking"])
def test_unbounded_decay_variants_remain_finite(core):
    q, k, v, _, b, w, S0 = _inputs(seed=3)
    # |G_C|=192 is beyond the centered fp32 factorization's documented range.
    g = jnp.full_like(k, -24.0)
    actual_o, actual_s = chunkwise_gated_delta_rule_2(
        q,
        k,
        v,
        g,
        b,
        w,
        S0,
        chunk_size=8,
        core=core,
        sub_chunk_size=2,
    )
    expected_o, expected_s = recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0)
    assert bool(jnp.all(jnp.isfinite(actual_o)))
    assert bool(jnp.all(jnp.isfinite(actual_s)))
    np.testing.assert_allclose(actual_o, expected_o, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual_s, expected_s, rtol=2e-5, atol=2e-6)
