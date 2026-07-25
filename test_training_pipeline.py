"""Regression tests for padding-aware training and checkpointing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from kimi_linear_gdn2 import KimiLinear
from test_architecture_features import tiny_config
from training import (
    TrainingConfig,
    create_optimizer,
    language_model_loss,
    make_lm_batch,
    restore_checkpoint,
    save_checkpoint,
    train_step,
)


def test_make_lm_batch_shifts_truncates_and_right_pads():
    batch = make_lm_batch(
        [[1, 2, 3, 4, 5], [6, 7, 8]],
        pad_token_id=31,
        max_seq_len=3,
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3], [6, 7, 31]]
    assert batch["labels"].tolist() == [[2, 3, 4], [7, 8, 31]]
    assert batch["attention_mask"].tolist() == [
        [True, True, True],
        [True, True, False],
    ]


@pytest.mark.parametrize("attnres_mode", ["none", "full", "block"])
def test_padding_does_not_change_valid_logits_or_moe_counts(attnres_mode):
    model = KimiLinear(
        tiny_config(attnres_mode=attnres_mode),
        rngs=nnx.Rngs(20),
    )
    batch = make_lm_batch([[1, 2, 3, 4], [5, 6]], pad_token_id=31)

    padded_logits, aux = model(
        batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    first_logits, _ = model(jnp.array([[1, 2, 3]], jnp.int32))
    second_logits, _ = model(jnp.array([[5]], jnp.int32))

    assert jnp.allclose(padded_logits[0, :3], first_logits[0], atol=3e-5, rtol=3e-5)
    assert jnp.allclose(padded_logits[1, :1], second_logits[0], atol=3e-5, rtol=3e-5)
    assert jnp.all(padded_logits[1, 1:] == 0)
    expected_assignments = int(batch["attention_mask"].sum()) * model.cfg.moe_top_k
    assert jnp.all(aux["group_sizes"].sum(axis=-1) == expected_assignments)


def test_masked_loss_has_finite_gradients_and_no_pad_embedding_gradient():
    model = KimiLinear(tiny_config(), rngs=nnx.Rngs(21))
    batch = make_lm_batch([[1, 2, 3, 4], [5, 6]], pad_token_id=31)

    def loss_fn(current_model):
        return language_model_loss(current_model, batch)[0]

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    leaves = jax.tree.leaves(grads)

    assert jnp.isfinite(loss)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert jnp.all(grads["embed"]["embedding"][31] == 0)


def test_train_step_updates_parameters_and_checkpoint_round_trips(tmp_path):
    config = tiny_config()
    training_config = TrainingConfig()
    batch = make_lm_batch([[1, 2, 3], [4, 5]], pad_token_id=31)
    model = KimiLinear(config, rngs=nnx.Rngs(22))
    optimizer = create_optimizer(model, training_config)
    before = model.embed.embedding[...].copy()

    metrics = train_step(
        model,
        optimizer,
        batch,
        aux_loss_weight=training_config.aux_loss_weight,
        router_bias_lr=training_config.router_bias_lr,
    )

    assert int(metrics["step"]) == 1
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(metrics["grad_norm"])
    assert not jnp.array_equal(before, model.embed.embedding[...])

    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model,
        optimizer,
        model_config=config,
        training_config=training_config,
    )
    restored_model = KimiLinear(config, rngs=nnx.Rngs(23))
    restored_optimizer = create_optimizer(restored_model, training_config)
    metadata = restore_checkpoint(
        checkpoint, restored_model, restored_optimizer
    )

    expected, _ = model(
        batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    actual, _ = restored_model(
        batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    assert metadata["format_version"] == 1
    assert int(restored_optimizer.step[...]) == 1
    assert jnp.array_equal(expected, actual)
