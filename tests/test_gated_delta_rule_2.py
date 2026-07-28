import jax
import jax.numpy as jnp
import pytest

from core import (
    gated_delta_rule_2_chunkwise as module_chunkwise,
    gated_delta_rule_2_recurrent as module_recurrent,
)
from gated_deltanet_2.core import (
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)


def _inputs(*, dtype=jnp.float32):
    batch, length, heads, key_dim, value_dim = 2, 8, 3, 4, 5
    keys = jax.random.split(jax.random.key(0), 7)
    q = jax.random.normal(keys[0], (batch, length, heads, key_dim), dtype=dtype)
    k = jax.random.normal(keys[1], (batch, length, heads, key_dim), dtype=dtype)
    v = jax.random.normal(keys[2], (batch, length, heads, value_dim), dtype=dtype)
    g = -jax.random.uniform(keys[3], q.shape, minval=0, maxval=0.1, dtype=dtype)
    b = jax.random.uniform(keys[4], q.shape, dtype=dtype)
    w = jax.random.uniform(keys[5], v.shape, dtype=dtype)
    state = jax.random.normal(
        keys[6], (batch, heads, key_dim, value_dim), dtype=dtype
    )
    return q, k, v, g, b, w, state


def test_chunkwise_matches_recurrent_forward_and_state():
    inputs = _inputs()
    recurrent_output, recurrent_state = recurrent_gated_delta_rule_2(*inputs)
    chunk_output, chunk_state = chunkwise_gated_delta_rule_2(
        *inputs, chunk_size=4
    )

    assert recurrent_output.shape == (2, 8, 3, 5)
    assert chunk_output.shape == recurrent_output.shape
    assert jnp.allclose(chunk_output, recurrent_output, rtol=1e-5, atol=2e-5)
    assert jnp.allclose(chunk_state, recurrent_state, rtol=1e-5, atol=2e-5)


def test_chunkwise_matches_recurrent_gradients():
    inputs = _inputs()

    def recurrent_loss(*args):
        output, state = recurrent_gated_delta_rule_2(*args)
        return output.sum() + state.sum()

    def chunk_loss(*args):
        output, state = chunkwise_gated_delta_rule_2(*args, chunk_size=4)
        return output.sum() + state.sum()

    argnums = tuple(range(7))
    recurrent_grads = jax.grad(recurrent_loss, argnums=argnums)(*inputs)
    chunk_grads = jax.grad(chunk_loss, argnums=argnums)(*inputs)

    for chunk_grad, recurrent_grad in zip(chunk_grads, recurrent_grads):
        assert jnp.allclose(
            chunk_grad, recurrent_grad, rtol=2e-5, atol=4e-5
        )


@pytest.mark.parametrize(
    "fn",
    [
        recurrent_gated_delta_rule_2,
        lambda *args: chunkwise_gated_delta_rule_2(*args, chunk_size=4),
        module_recurrent,
        lambda *args: module_chunkwise(*args, chunk_size=4),
    ],
    ids=[
        "package-recurrent",
        "package-chunkwise",
        "module-recurrent",
        "module-chunkwise",
    ],
)
def test_output_preserves_model_dtype_and_state_is_fp32(fn):
    output, state = fn(*_inputs(dtype=jnp.bfloat16))

    assert output.dtype == jnp.bfloat16
    assert state.dtype == jnp.float32


def test_rejects_non_paper_layout():
    inputs = _inputs()
    head_major = tuple(jnp.swapaxes(x, 1, 2) for x in inputs[:6])

    with pytest.raises(ValueError, match="S0 must have shape"):
        recurrent_gated_delta_rule_2(*head_major, inputs[6])
