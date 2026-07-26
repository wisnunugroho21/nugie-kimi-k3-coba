"""Integration checks for the paper-aligned Gated DeltaNet-2 token mixer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from gated_deltanet_2.layer import GatedDeltaNet2, LowRankLinear


def _layer(seed: int = 0) -> GatedDeltaNet2:
    return GatedDeltaNet2(
        d_model=24,
        num_heads=2,
        head_k_dim=4,
        head_v_dim=4,
        num_v_heads=4,
        chunk_size=4,
        conv_size=4,
        rngs=nnx.Rngs(seed),
    )


def test_ragged_full_forward_matches_token_streaming_with_gqa():
    layer = _layer()
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 24))

    expected_o, expected_s = layer(x, return_state=True)

    cache = layer.init_cache(batch_size=2)
    streamed = []
    for token in range(x.shape[1]):
        token_o, cache = layer.step(x[:, token : token + 1], cache)
        streamed.append(token_o)
    actual_o = jnp.concatenate(streamed, axis=1)

    np.testing.assert_allclose(actual_o, expected_o, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        cache.recurrent_state,
        expected_s,
        rtol=2e-5,
        atol=2e-6,
    )


def test_folded_gqa_matches_explicit_key_side_repetition():
    layer = _layer(seed=7)
    x = jax.random.normal(jax.random.PRNGKey(8), (1, 6, 24))
    q, k, v, g, b, w, _ = layer._project(x, conv_states=None)
    public_state = jax.random.normal(
        jax.random.PRNGKey(9),
        (1, layer.Hv, layer.dk, layer.dv),
    )

    folded_o, folded_s = layer._run_recurrence(
        q,
        k,
        v,
        g,
        b,
        w,
        layer._state_in(public_state),
    )

    def repeat_key_side(tensor):
        return jnp.repeat(tensor, layer.group, axis=1)

    def expand_value_side(tensor):
        B, H, L, _ = tensor.shape
        return (
            tensor.reshape(B, H, L, layer.group, layer.dv)
            .transpose(0, 1, 3, 2, 4)
            .reshape(B, layer.Hv, L, layer.dv)
        )

    repeated_o, repeated_s = layer._run_recurrence(
        repeat_key_side(q),
        repeat_key_side(k),
        expand_value_side(v),
        repeat_key_side(g),
        repeat_key_side(b),
        expand_value_side(w),
        public_state,
    )

    np.testing.assert_allclose(
        expand_value_side(folded_o),
        repeated_o,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        layer._state_out(folded_s),
        repeated_s,
        rtol=2e-5,
        atol=2e-6,
    )


def test_attention_mask_keeps_padding_out_of_output_and_state():
    layer = _layer(seed=2)
    x = jax.random.normal(jax.random.PRNGKey(3), (1, 5, 24))
    expected_o, expected_s = layer(x, return_state=True)

    padded_x = jnp.pad(x, ((0, 0), (0, 3), (0, 0)), constant_values=7.0)
    mask = jnp.array([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=jnp.int32)
    actual_o, actual_s = layer(
        padded_x,
        attention_mask=mask,
        return_state=True,
    )

    np.testing.assert_allclose(actual_o[:, :5], expected_o, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual_o[:, 5:], jnp.zeros_like(actual_o[:, 5:]))
    np.testing.assert_allclose(actual_s, expected_s, rtol=2e-5, atol=2e-6)


def test_reference_projection_parameterization():
    layer = _layer(seed=4)

    assert layer.b_proj.bias is None
    assert layer.w_proj.bias is None
    assert isinstance(layer.f_proj, LowRankLinear)
    assert layer.f_proj.down.bias is None
    assert layer.f_proj.up.bias is None
    assert isinstance(layer.o_norm.gate, LowRankLinear)
    assert layer.o_norm.gate.down.bias is None
    assert layer.o_norm.gate.up.bias is not None
    assert layer.q_conv.conv.bias is None
    assert layer.k_conv.conv.bias is None
    assert layer.v_conv.conv.bias is None


def test_invalid_mask_shape_is_rejected():
    layer = _layer(seed=5)
    x = jnp.ones((2, 3, 24))
    with pytest.raises(ValueError, match="attention_mask must have shape"):
        layer(x, attention_mask=jnp.ones((2, 2)))


def test_empty_sequence_is_rejected():
    layer = _layer(seed=6)
    with pytest.raises(ValueError, match="non-empty sequence"):
        layer(jnp.empty((1, 0, 24)))
    with pytest.raises(ValueError, match="non-empty sequence"):
        layer.step(jnp.empty((1, 0, 24)), layer.init_cache(1))


@pytest.mark.parametrize(
    "override",
    [
        {"num_heads": 0},
        {"num_v_heads": 0},
        {"num_v_heads": 3},
        {"chunk_size": 0},
        {"conv_size": 0},
        {"sub_chunk_size": 0},
    ],
)
def test_invalid_layer_configuration_is_rejected(override):
    kwargs = {
        "d_model": 24,
        "num_heads": 2,
        "head_k_dim": 4,
        "head_v_dim": 4,
        "num_v_heads": 4,
        "chunk_size": 4,
        "conv_size": 4,
        "sub_chunk_size": 2,
    }
    kwargs.update(override)
    with pytest.raises(ValueError):
        GatedDeltaNet2(**kwargs, rngs=nnx.Rngs(10))
