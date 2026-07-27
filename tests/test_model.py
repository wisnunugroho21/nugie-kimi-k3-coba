"""Focused behavioral tests for the educational Kimi K3."""

import jax
import jax.numpy as jnp
from flax import nnx

from kimi_k3 import KimiDeltaAttention, KimiK3, KimiK3Config, causal_lm_loss
from kimi_k3.model import QuantileRouter


def small_config(**overrides: object) -> KimiK3Config:
    """Return a particularly small model so tests stay quick on CPU."""
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_layers": 5,
        "num_heads": 2,
        "head_dim": 8,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "dense_hidden_size": 24,
        "latent_moe_dim": 8,
        "moe_hidden_size": 12,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "attn_res_block_size": 4,
        "max_sequence_length": 32,
    }
    values.update(overrides)
    return KimiK3Config.tiny(**values)


def test_paper_attention_schedule() -> None:
    """K3 uses 69 KDA layers and 24 MLA layers, including final MLA."""
    config = KimiK3Config.paper()
    kinds = [config.attention_kind(i) for i in range(config.num_layers)]
    assert kinds.count("kda") == 69
    assert kinds.count("mla") == 24
    assert kinds[-1] == "mla"


def test_forward_shape_is_finite_and_differentiable() -> None:
    """The complete model returns logits and supports NNX gradients."""
    config = small_config()
    model = KimiK3(config, rngs=nnx.Rngs(0))
    tokens = jnp.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    logits = model(tokens)

    assert logits.shape == (2, 5, config.vocab_size)
    assert bool(jnp.all(jnp.isfinite(logits)))

    gradients = nnx.grad(lambda m: causal_lm_loss(m(tokens), tokens))(model)
    leaves = jax.tree.leaves(gradients)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_model_is_causal() -> None:
    """Changing a suffix cannot alter logits produced for an equal prefix."""
    config = small_config()
    model = KimiK3(config, rngs=nnx.Rngs(1))
    first = jnp.array([[1, 2, 3, 4, 5, 6]])
    second = jnp.array([[1, 2, 3, 9, 8, 7]])

    first_logits = model(first)
    second_logits = model(second)
    assert bool(jnp.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-5))


def test_kda_retention_is_strictly_lower_bounded() -> None:
    """Equation (5) keeps exp(g_min) < alpha < 1 for every channel."""
    config = small_config()
    kda = KimiDeltaAttention(config, rngs=nnx.Rngs(2))
    x = jax.random.normal(jax.random.key(3), (2, 7, config.hidden_size))
    alpha = kda.retention(x)

    assert bool(jnp.all(alpha > jnp.exp(config.gate_lower_bound)))
    assert bool(jnp.all(alpha < 1.0))


def test_quantile_router_uses_centered_non_gradient_bias() -> None:
    """Quantile Balancing updates centered state and leaves weights normalized."""
    config = small_config()
    router = QuantileRouter(config, rngs=nnx.Rngs(4))
    x = jax.random.normal(jax.random.key(5), (3, 6, config.hidden_size))

    indices, weights, raw_scores = router(x)
    new_bias = router.update_bias(raw_scores)

    assert indices.shape == (3, 6, config.num_experts_per_token)
    assert bool(jnp.allclose(weights.sum(axis=-1), 1.0))
    assert bool(jnp.allclose(new_bias.mean(), 0.0, atol=1e-6))
    assert bool(jnp.allclose(router.correction_bias[...], new_bias))
