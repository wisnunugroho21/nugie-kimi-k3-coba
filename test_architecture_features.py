"""Focused regression tests for the Kimi/GDN-2 architecture extensions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from attention_residual import AttentionResidual
from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig
from multi_latent_attention.attention import (
    GatedMultiHeadLatentAttention,
    GroupedQueryLatentAttention,
)
from multi_latent_attention.moe import LatentMoE


def tiny_config(**overrides) -> KimiLinearConfig:
    values = dict(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        full_attn_period=2,
        gdn_num_heads=2,
        gdn_head_k_dim=4,
        gdn_head_v_dim=4,
        gdn_chunk_size=2,
        gdn_conv_size=2,
        mla_num_q_heads=2,
        mla_num_kv_heads=1,
        mla_head_dim=4,
        max_seq_len=8,
        moe_d_ff=12,
        moe_n_routed=4,
        moe_n_shared=1,
        moe_top_k=2,
        moe_n_groups=2,
        moe_topk_groups=1,
        moe_latent_dim=4,
        attnres_mode="block",
        attnres_block_size=2,
    )
    values.update(overrides)
    return KimiLinearConfig(**values)


def test_attnres_zero_query_starts_as_uniform_average():
    residual = AttentionResidual(3, rngs=nnx.Rngs(0))
    values = [
        jnp.ones((1, 2, 3)),
        2.0 * jnp.ones((1, 2, 3)),
        6.0 * jnp.ones((1, 2, 3)),
    ]

    output, weights = residual(values, return_weights=True)

    assert jnp.allclose(weights, jnp.full((3, 1, 2), 1.0 / 3.0))
    assert jnp.allclose(output, 3.0 * jnp.ones((1, 2, 3)))


def test_latent_moe_sparse_dispatch_matches_dense_reference():
    moe = LatentMoE(
        d_model=8,
        latent_dim=3,
        d_ff=6,
        n_routed=4,
        n_shared=1,
        top_k=2,
        n_groups=2,
        topk_groups=1,
        rngs=nnx.Rngs(1),
    )
    x = jax.random.normal(jax.random.key(2), (2, 3, 8))

    sparse, aux = moe(x)
    dense = moe.dense_forward(x)

    assert sparse.shape == x.shape
    assert moe.w_in[...].shape == (4, 3, 12)
    assert moe.w_out[...].shape == (4, 6, 3)
    assert jnp.allclose(sparse, dense, rtol=2e-5, atol=2e-5)
    assert int(aux["group_sizes"].sum()) == 2 * 3 * 2


def test_gated_mla_full_and_cached_prefill_match():
    mla = GatedMultiHeadLatentAttention(
        embed_dim=12,
        num_q_heads=3,
        num_kv_heads=1,
        head_dim=4,
        rngs=nnx.Rngs(3),
    )
    x = jax.random.normal(jax.random.key(4), (2, 5, 12))

    full = mla(x)
    cached, cache = mla.step(x, mla.init_cache(batch_size=2, max_len=7))

    assert mla.gate_proj is not None
    assert mla.gate_proj.kernel[...].shape == (12, 12)
    assert jnp.allclose(full, cached, rtol=1e-5, atol=1e-5)
    assert int(cache.pos) == 5


def test_gated_mla_bfloat_cache_uses_compute_dtype_and_stays_equivalent():
    mla = GatedMultiHeadLatentAttention(
        embed_dim=12,
        num_q_heads=3,
        num_kv_heads=1,
        head_dim=4,
        compute_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(9),
    )
    x = jax.random.normal(jax.random.key(10), (1, 5, 12))
    cache = mla.init_cache(batch_size=1, max_len=7)

    full = mla(x)
    cached, _ = mla.step(x, cache)

    assert cache.l_kv.dtype == jnp.bfloat16
    assert jnp.allclose(full, cached, rtol=2e-2, atol=2e-3)


def test_gated_mla_rejects_cache_overflow():
    mla = GatedMultiHeadLatentAttention(
        embed_dim=8,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        rngs=nnx.Rngs(11),
    )
    x = jnp.ones((1, 3, 8))

    with pytest.raises(ValueError, match="exceeds MLA cache capacity"):
        mla.step(x, mla.init_cache(batch_size=1, max_len=2))


def test_gated_mla_gate_is_applied_before_linear_output_projection():
    kwargs = dict(
        embed_dim=12,
        num_q_heads=3,
        num_kv_heads=1,
        head_dim=4,
    )
    gated = GatedMultiHeadLatentAttention(**kwargs, rngs=nnx.Rngs(7))
    ungated = GroupedQueryLatentAttention(**kwargs, rngs=nnx.Rngs(7))
    x = jax.random.normal(jax.random.key(8), (1, 3, 12))

    # A zero gate projection gives sigmoid(0)=0.5 for every head channel. Since
    # the following absorbed output projection is linear and bias-free, the
    # complete gated output must be exactly half the ungated output.
    assert gated.gate_proj is not None
    gated.gate_proj.kernel[...] = 0.0

    assert jnp.allclose(gated(x), 0.5 * ungated(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("attnres_mode", ["none", "full", "block"])
def test_model_full_and_cached_prefill_match(attnres_mode):
    model = KimiLinear(
        tiny_config(attnres_mode=attnres_mode),
        rngs=nnx.Rngs(5),
    )
    # Length five deliberately leaves a ragged token after two-token GDN chunks.
    ids = jnp.array([[1, 2, 3, 4, 5]], jnp.int32)

    full, aux = model(ids)
    cached, caches = model.step(ids, model.init_cache(batch_size=1, max_len=8))

    assert full.shape == (1, 5, 32)
    assert aux["group_sizes"].shape == (2, 4)
    assert len(caches) == 2
    assert jnp.all(jnp.isfinite(full))
    assert jnp.allclose(full, cached, rtol=2e-5, atol=2e-5)


def test_nonzero_block_attnres_matches_token_by_token_streaming():
    model = KimiLinear(
        tiny_config(attnres_mode="block", attnres_block_size=3),
        rngs=nnx.Rngs(12),
    )
    residuals = []
    for layer in model.layers:
        assert layer.token_residual is not None
        assert layer.channel_residual is not None
        residuals.extend((layer.token_residual, layer.channel_residual))
    assert model.final_residual is not None
    residuals.append(model.final_residual)
    for index, residual in enumerate(residuals, start=1):
        residual.query[...] = index * jnp.linspace(-0.1, 0.1, 16)

    ids = jnp.array([[1, 2, 3, 4, 5]], jnp.int32)
    full, _ = model(ids)
    caches = model.init_cache(batch_size=1, max_len=8)
    outputs = []
    for position in range(ids.shape[1]):
        output, caches = model.step(ids[:, position : position + 1], caches)
        outputs.append(output)
    streamed = jnp.concatenate(outputs, axis=1)

    assert jnp.allclose(full, streamed, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"n_layers": 0}, "n_layers must be positive"),
        ({"gdn_chunk_size": 0}, "gdn_chunk_size must be positive"),
        ({"mla_num_kv_heads": 0}, "mla_num_kv_heads must be positive"),
        ({"moe_top_k": 0}, "moe_top_k must be positive"),
        ({"compute_dtype": "float16"}, "compute_dtype"),
    ],
)
def test_invalid_configurations_fail_early(override, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**override)


def test_generation_validates_token_count_and_cache_capacity():
    model = KimiLinear(tiny_config(), rngs=nnx.Rngs(13))
    prompt = jnp.array([[1, 2]], jnp.int32)

    assert model.generate(prompt, 0).shape == (1, 0)
    with pytest.raises(ValueError, match="cannot be negative"):
        model.generate(prompt, -1)
    with pytest.raises(ValueError, match="too small"):
        model.generate(prompt, 3, max_len=3)


def test_all_new_parameters_receive_finite_gradients():
    model = KimiLinear(tiny_config(), rngs=nnx.Rngs(6))
    ids = jnp.array([[1, 2, 3]], jnp.int32)

    def loss_fn(m):
        logits, aux = m(ids)
        return jnp.mean(logits**2) + aux["aux_loss"]

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    grad_leaves = jax.tree.leaves(grads)

    assert jnp.isfinite(loss)
    assert grad_leaves
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves)
