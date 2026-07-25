"""Focused regression tests for the Kimi/GDN-2 architecture extensions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from attention_residual import (
    AttentionResidual,
    AttentionResidualState,
    prepare_batched_attention_residual,
)
from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig
from multi_latent_attention.attention import (
    GatedMultiHeadLatentAttention,
    GroupedQueryLatentAttention,
)
from multi_latent_attention.moe import LatentMoE


def tiny_config(**overrides) -> KimiLinearConfig:
    values = {
        "vocab_size": 32,
        "d_model": 16,
        "n_layers": 2,
        "full_attn_period": 2,
        "gdn_num_heads": 2,
        "gdn_head_k_dim": 4,
        "gdn_head_v_dim": 4,
        "gdn_chunk_size": 2,
        "gdn_conv_size": 2,
        "mla_num_q_heads": 2,
        "mla_num_kv_heads": 1,
        "mla_head_dim": 4,
        "max_seq_len": 8,
        "moe_d_ff": 12,
        "moe_design_mode": "custom",
        "moe_n_routed": 4,
        "moe_n_shared": 1,
        "moe_top_k": 2,
        "moe_n_groups": 2,
        "moe_topk_groups": 1,
        "moe_latent_dim": 4,
        "attnres_mode": "block",
        "attnres_block_size": 2,
    }
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


def test_cached_attnres_matches_original_rmsnorm_formulation():
    residual = AttentionResidual(5, rngs=nnx.Rngs(29))
    residual.query[...] = jnp.linspace(-0.4, 0.5, 5)
    residual.key_norm.weight[...] = jnp.linspace(0.7, 1.3, 5)
    values = [jax.random.normal(jax.random.key(index), (2, 3, 5)) for index in range(3)]
    partial = jax.random.normal(jax.random.key(30), (2, 3, 5))
    stacked = jnp.stack((*values, partial), axis=0)

    keys = residual.key_norm(stacked).astype(jnp.float32)
    logits = jnp.einsum("d,sbtd->sbt", residual.query[...].astype(jnp.float32), keys)
    expected_weights = jax.nn.softmax(logits, axis=0)
    expected = jnp.einsum("sbt,sbtd->btd", expected_weights, stacked)

    state = AttentionResidualState.initialize(values[0], eps=residual.key_norm.eps)
    for value in values[1:]:
        state = state.append(value, eps=residual.key_norm.eps)
    actual, actual_weights = residual(state, partial=partial, return_weights=True)

    assert state.values.shape == state.normalized.shape == (3, 2, 3, 5)
    assert jnp.allclose(actual_weights, expected_weights, atol=2e-6, rtol=2e-6)
    assert jnp.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_two_phase_attnres_matches_independent_softmax_and_gradients():
    residuals = [AttentionResidual(4, rngs=nnx.Rngs(31 + index)) for index in range(3)]
    for index, residual in enumerate(residuals, start=1):
        residual.query[...] = index * jnp.linspace(-0.2, 0.3, 4)
        residual.key_norm.weight[...] = jnp.linspace(0.8, 1.2, 4)

    completed = [
        jax.random.normal(jax.random.key(35 + index), (1, 3, 4)) for index in range(3)
    ]
    partials = [
        None,
        jax.random.normal(jax.random.key(38), (1, 3, 4)),
        jax.random.normal(jax.random.key(39), (1, 3, 4)),
    ]

    state = AttentionResidualState.initialize(completed[0], eps=1e-5)
    for value in completed[1:]:
        state = state.append(value, eps=1e-5)
    phase = prepare_batched_attention_residual(residuals, state)

    for index, (residual, partial) in enumerate(zip(residuals, partials)):
        expected_values = completed if partial is None else [*completed, partial]
        expected = residual(expected_values)
        actual = residual.merge_phase(phase, index, partial)
        assert jnp.allclose(actual, expected, atol=2e-6, rtol=2e-6)

    def direct_loss(partial):
        return jnp.sum(residuals[1]([*completed, partial]) ** 2)

    def phased_loss(partial):
        current_phase = prepare_batched_attention_residual(residuals, state)
        return jnp.sum(residuals[1].merge_phase(current_phase, 1, partial) ** 2)

    direct_grad = jax.grad(direct_loss)(partials[1])
    phased_grad = jax.grad(phased_loss)(partials[1])
    assert jnp.allclose(phased_grad, direct_grad, atol=3e-5, rtol=3e-5)

    def direct_source_loss(first_source):
        return jnp.sum(residuals[1]([first_source, *completed[1:], partials[1]]) ** 2)

    def phased_source_loss(first_source):
        current_state = AttentionResidualState.initialize(first_source, eps=1e-5)
        for value in completed[1:]:
            current_state = current_state.append(value, eps=1e-5)
        current_phase = prepare_batched_attention_residual(residuals, current_state)
        return jnp.sum(residuals[1].merge_phase(current_phase, 1, partials[1]) ** 2)

    direct_source_grad = jax.grad(direct_source_loss)(completed[0])
    phased_source_grad = jax.grad(phased_source_loss)(completed[0])
    assert jnp.allclose(phased_source_grad, direct_source_grad, atol=3e-5, rtol=3e-5)


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


@pytest.mark.parametrize(
    "mode, expected_experts, expected_top_k",
    [
        ("efficiency", 16, 2),
        ("accuracy", 16, 8),
        ("custom", 4, 2),
    ],
)
def test_latent_moe_design_modes_resolve_counts(mode, expected_experts, expected_top_k):
    config = tiny_config(moe_design_mode=mode)
    model = KimiLinear(config, rngs=nnx.Rngs(27))

    assert config.moe_compression_ratio == 4
    assert config.moe_effective_n_routed == expected_experts
    assert config.moe_effective_top_k == expected_top_k
    for layer in model.layers:
        assert layer.channel_mixer.E == expected_experts
        assert layer.channel_mixer.top_k == expected_top_k


def test_latent_moe_design_report_exposes_cost_tradeoff():
    default = KimiLinearConfig()
    efficiency_config = tiny_config(moe_design_mode="efficiency")
    accuracy_config = tiny_config(moe_design_mode="accuracy")
    efficiency = efficiency_config.moe_design_report()
    accuracy = accuracy_config.moe_design_report()
    custom = tiny_config(moe_design_mode="custom").moe_design_report()
    model = KimiLinear(accuracy_config, rngs=nnx.Rngs(28))
    actual_params = sum(
        leaf.size
        for leaf in jax.tree.leaves(nnx.state(model.layers[0].channel_mixer, nnx.Param))
    )

    assert default.moe_design_mode == "accuracy"
    assert default.moe_effective_n_routed == 32
    assert default.moe_effective_top_k == 8
    assert efficiency["effective_n_routed"] == accuracy["effective_n_routed"] == 16
    assert efficiency["effective_top_k"] == 2
    assert accuracy["effective_top_k"] == 8
    assert (
        efficiency["estimated_parameters_per_layer"]
        == accuracy["estimated_parameters_per_layer"]
    )
    assert (
        accuracy["estimated_flops_per_token_per_layer"]
        > efficiency["estimated_flops_per_token_per_layer"]
    )
    assert (
        custom["estimated_parameters_per_layer"]
        < efficiency["estimated_parameters_per_layer"]
    )
    assert accuracy["estimated_parameters_per_layer"] == actual_params
    assert (
        accuracy["estimated_parameters_all_moe_layers"]
        == actual_params * accuracy_config.n_layers
    )


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


def _repeated_kv_mla_reference(mla, x, attention_mask, segment_ids):
    """Pre-optimization MLA math with explicitly repeated KV heads."""
    valid = attention_mask.astype(bool)
    masked_x = jnp.where(valid[..., None], x, 0)
    batch_size, seq_length, _ = x.shape
    q = (
        mla.w_q_uk(masked_x)
        .reshape(batch_size, seq_length, mla.num_q_heads, mla.head_dim)
        .transpose(0, 2, 1, 3)
    )
    kv = (
        mla.w_dkv(masked_x)
        .reshape(batch_size, seq_length, mla.num_kv_heads, mla.head_dim)
        .transpose(0, 2, 1, 3)
    )
    repeated_kv = kv.repeat(mla.group_size, axis=1)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, repeated_kv).astype(
        jnp.float32
    ) / jnp.sqrt(mla.head_dim)
    mask = (
        jnp.tril(jnp.ones((seq_length, seq_length), dtype=bool))[None]
        & valid[:, :, None]
        & valid[:, None, :]
    )
    previous_valid = jnp.pad(valid[:, :-1], ((0, 0), (1, 0)))
    previous_segment = jnp.pad(segment_ids[:, :-1], ((0, 0), (1, 0)))
    segment_start = valid & ((~previous_valid) | (segment_ids != previous_segment))
    segment_run = jnp.cumsum(segment_start, axis=1)
    mask &= segment_run[:, :, None] == segment_run[:, None, :]
    diagonal = jnp.eye(seq_length, dtype=bool)[None]
    mask |= (~valid)[:, :, None] & diagonal
    probabilities = jax.nn.softmax(
        jnp.where(mask[:, None], logits, -jnp.inf), axis=-1
    ).astype(repeated_kv.dtype)
    weighted = (
        jnp.einsum("bhqk,bhkd->bhqd", probabilities, repeated_kv)
        .transpose(0, 2, 1, 3)
        .reshape(batch_size, seq_length, mla.num_q_heads * mla.head_dim)
    )
    output = mla._output(weighted, masked_x)
    return jnp.where(valid[..., None], output, 0)


def test_mla_chunked_grouped_attention_matches_repeated_kv_reference():
    mla = GroupedQueryLatentAttention(
        embed_dim=12,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=3,
        query_chunk_size=2,
        rngs=nnx.Rngs(101),
    )
    x = jax.random.normal(jax.random.key(102), (2, 5, 12))
    attention_mask = jnp.array([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], dtype=bool)
    # Reusing raw ID 1 after a boundary deliberately exercises contiguous-run
    # canonicalization in addition to query chunk boundaries.
    segment_ids = jnp.array([[1, 1, 2, 2, 1], [9, 1, 1, 2, 2]], jnp.int32)

    actual = mla(x, attention_mask, segment_ids)
    expected = _repeated_kv_mla_reference(mla, x, attention_mask, segment_ids)

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    gradient = jax.grad(
        lambda inputs: jnp.sum(mla(inputs, attention_mask, segment_ids))
    )(x)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(gradient[~attention_mask] == 0)


def test_mla_paged_decode_matches_full_attention_across_partial_pages():
    mla = GatedMultiHeadLatentAttention(
        embed_dim=12,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=3,
        query_chunk_size=3,
        cache_page_size=2,
        rngs=nnx.Rngs(103),
    )
    x = jax.random.normal(jax.random.key(104), (1, 7, 12))
    full = mla(x)
    cache = mla.init_cache(batch_size=1, max_len=9)

    outputs = []
    for start, stop in ((0, 3), (3, 5), (5, 6), (6, 7)):
        output, cache = mla.step(x[:, start:stop], cache)
        outputs.append(output)

    streamed = jnp.concatenate(outputs, axis=1)
    assert jnp.allclose(full, streamed, rtol=1e-5, atol=1e-5)
    assert int(cache.pos) == 7
    assert cache.l_kv.shape == (1, 9, 6)


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
        ({"mla_query_chunk_size": 0}, "mla_query_chunk_size must be positive"),
        ({"mla_cache_page_size": 0}, "mla_cache_page_size must be positive"),
        ({"moe_top_k": 0}, "moe_top_k must be positive"),
        ({"moe_design_mode": "fast"}, "moe_design_mode"),
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


def test_generation_runs_jitted_two_phase_attnres_decode():
    model = KimiLinear(tiny_config(), rngs=nnx.Rngs(40))
    generated = model.generate(
        jnp.array([[1, 2, 3]], jnp.int32),
        max_new_tokens=2,
        max_len=5,
    )

    assert generated.shape == (1, 2)
    assert jnp.issubdtype(generated.dtype, jnp.integer)


def test_latent_moe_presets_require_integer_compression_ratio():
    with pytest.raises(ValueError, match="alpha is an integer"):
        tiny_config(
            d_model=15,
            moe_latent_dim=4,
            moe_design_mode="accuracy",
        )


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
