"""Padding- and packing-aware training utilities for :mod:`kimi_linear_gdn2`.

The public pipeline is intentionally small:

    batch = make_lm_batch(token_sequences, pad_token_id=0)
    optimizer = create_optimizer(model, TrainingConfig())
    metrics = train_step(model, optimizer, batch)

`make_lm_batch` right-pads raw sequences; `make_packed_lm_batch` greedily combines
documents and emits segment IDs. The masks are enforced inside the model as well
as at the loss, so neither padding nor earlier packed documents leak into a token.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig
from multi_latent_attention.moe import update_router_bias


class LMBatch(TypedDict):
    """A padded or packed next-token-prediction batch."""

    input_ids: jax.Array
    labels: jax.Array
    attention_mask: jax.Array
    segment_ids: NotRequired[jax.Array]


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    aux_loss_weight: float = 1.0
    router_bias_lr: float = 1e-3

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.aux_loss_weight < 0:
            raise ValueError("aux_loss_weight cannot be negative")
        if self.router_bias_lr < 0:
            raise ValueError("router_bias_lr cannot be negative")


def make_lm_batch(
    token_sequences: Sequence[Sequence[int] | np.ndarray | jax.Array],
    *,
    pad_token_id: int,
    max_seq_len: int | None = None,
    dtype: jnp.dtype = jnp.int32,
) -> LMBatch:
    """Shift and right-pad raw token sequences for causal LM training.

    Every source sequence must contain at least two tokens. If ``max_seq_len`` is
    supplied, at most ``max_seq_len + 1`` source tokens are retained so the
    resulting input and label arrays fit the model context limit.
    """
    if not token_sequences:
        raise ValueError("token_sequences cannot be empty")
    if max_seq_len is not None and max_seq_len < 1:
        raise ValueError("max_seq_len must be positive or None")

    sequences: list[np.ndarray] = []
    for index, sequence in enumerate(token_sequences):
        array = np.asarray(sequence)
        if array.ndim != 1:
            raise ValueError(f"sequence {index} must be one-dimensional")
        if array.size < 2:
            raise ValueError(
                f"sequence {index} needs at least two tokens for next-token training"
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"sequence {index} must contain integer token ids")
        if max_seq_len is not None:
            array = array[: max_seq_len + 1]
        sequences.append(array.astype(np.int32, copy=False))

    batch_size = len(sequences)
    sequence_length = max(sequence.size - 1 for sequence in sequences)
    input_ids = np.full(
        (batch_size, sequence_length), pad_token_id, dtype=np.int32
    )
    labels = np.full((batch_size, sequence_length), pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((batch_size, sequence_length), dtype=np.bool_)

    for row, sequence in enumerate(sequences):
        length = sequence.size - 1
        input_ids[row, :length] = sequence[:-1]
        labels[row, :length] = sequence[1:]
        attention_mask[row, :length] = True

    return {
        "input_ids": jnp.asarray(input_ids, dtype=dtype),
        "labels": jnp.asarray(labels, dtype=dtype),
        "attention_mask": jnp.asarray(attention_mask),
    }


def make_packed_lm_batch(
    token_sequences: Sequence[Sequence[int] | np.ndarray | jax.Array],
    *,
    pad_token_id: int,
    max_seq_len: int,
    dtype: jnp.dtype = jnp.int32,
) -> LMBatch:
    """Greedily pack independent sequences into fixed-length training rows.

    Next-token pairs are formed *before* packing, so the final token of one
    document is never trained to predict the first token of the next. Sequences
    longer than the context are truncated to ``max_seq_len + 1`` source tokens;
    shorter sequences are packed whole and never split across rows.
    """
    if not token_sequences:
        raise ValueError("token_sequences cannot be empty")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")

    examples: list[np.ndarray] = []
    for index, sequence in enumerate(token_sequences):
        array = np.asarray(sequence)
        if array.ndim != 1:
            raise ValueError(f"sequence {index} must be one-dimensional")
        if array.size < 2:
            raise ValueError(
                f"sequence {index} needs at least two tokens for next-token training"
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"sequence {index} must contain integer token ids")
        examples.append(
            array[: max_seq_len + 1].astype(np.int32, copy=False)
        )

    packed_rows: list[list[np.ndarray]] = []
    current_row: list[np.ndarray] = []
    current_length = 0
    for example in examples:
        pair_count = example.size - 1
        if current_row and current_length + pair_count > max_seq_len:
            packed_rows.append(current_row)
            current_row = []
            current_length = 0
        current_row.append(example)
        current_length += pair_count
    if current_row:
        packed_rows.append(current_row)

    batch_size = len(packed_rows)
    input_ids = np.full(
        (batch_size, max_seq_len), pad_token_id, dtype=np.int32
    )
    labels = np.full(
        (batch_size, max_seq_len), pad_token_id, dtype=np.int32
    )
    attention_mask = np.zeros((batch_size, max_seq_len), dtype=np.bool_)
    segment_ids = np.full((batch_size, max_seq_len), -1, dtype=np.int32)

    for row_index, row in enumerate(packed_rows):
        offset = 0
        for segment_id, example in enumerate(row):
            pair_count = example.size - 1
            end = offset + pair_count
            input_ids[row_index, offset:end] = example[:-1]
            labels[row_index, offset:end] = example[1:]
            attention_mask[row_index, offset:end] = True
            segment_ids[row_index, offset:end] = segment_id
            offset = end

    return {
        "input_ids": jnp.asarray(input_ids, dtype=dtype),
        "labels": jnp.asarray(labels, dtype=dtype),
        "attention_mask": jnp.asarray(attention_mask),
        "segment_ids": jnp.asarray(segment_ids),
    }


def create_optimizer(
    model: KimiLinear, config: TrainingConfig
) -> nnx.Optimizer:
    """Create an AdamW optimizer with global-norm gradient clipping."""
    transformation = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(
            learning_rate=config.learning_rate,
            b1=config.adam_beta1,
            b2=config.adam_beta2,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        ),
    )
    return nnx.Optimizer(model, transformation, wrt=nnx.Param)


def language_model_loss(
    model: KimiLinear,
    batch: LMBatch,
    *,
    aux_loss_weight: float | jax.Array = 1.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute padding-masked next-token cross entropy and MoE auxiliary loss."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    segment_ids = batch.get("segment_ids")
    if labels.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids, labels, and attention_mask must have equal shapes")
    if segment_ids is not None and segment_ids.shape != input_ids.shape:
        raise ValueError("segment_ids must have the same shape as input_ids")

    logits, aux = model(
        input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )
    valid = attention_mask.astype(bool)
    safe_labels = jnp.where(valid, labels, 0)
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, safe_labels
    )
    weights = valid.astype(jnp.float32)
    token_count = weights.sum()
    denominator = jnp.maximum(token_count, 1.0)
    cross_entropy = (per_token_loss * weights).sum() / denominator
    auxiliary_loss = aux["aux_loss"].astype(jnp.float32)
    total_loss = cross_entropy + aux_loss_weight * auxiliary_loss

    predictions = jnp.argmax(logits, axis=-1)
    accuracy = ((predictions == labels) * valid).sum() / denominator
    metrics = {
        "loss": total_loss,
        "cross_entropy": cross_entropy,
        "aux_loss": auxiliary_loss,
        "accuracy": accuracy,
        "perplexity": jnp.exp(jnp.minimum(cross_entropy, 20.0)),
        "token_count": token_count,
        "group_sizes": aux["group_sizes"],
    }
    return total_loss, metrics


def _apply_router_bias_updates(
    model: KimiLinear, group_sizes: jax.Array, learning_rate: float | jax.Array
) -> None:
    for layer_index, layer in enumerate(model.layers):
        router = layer.channel_mixer
        router.router_bias[...] = update_router_bias(
            router.router_bias[...], group_sizes[layer_index], lr=learning_rate
        )


@nnx.jit
def train_step(
    model: KimiLinear,
    optimizer: nnx.Optimizer,
    batch: LMBatch,
    *,
    aux_loss_weight: float | jax.Array = 1.0,
    router_bias_lr: float | jax.Array = 1e-3,
) -> dict[str, jax.Array]:
    """Run one compiled optimizer step and update aux-loss-free router biases."""

    def loss_fn(current_model: KimiLinear):
        return language_model_loss(
            current_model, batch, aux_loss_weight=aux_loss_weight
        )

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grad_norm = optax.tree.norm(grads)
    optimizer.update(model, grads)
    _apply_router_bias_updates(model, metrics["group_sizes"], router_bias_lr)

    result = dict(metrics)
    result["loss"] = loss
    result["grad_norm"] = grad_norm
    result["step"] = optimizer.step[...]
    return result


@nnx.jit
def eval_step(
    model: KimiLinear,
    batch: LMBatch,
    *,
    aux_loss_weight: float | jax.Array = 1.0,
) -> dict[str, jax.Array]:
    """Run one compiled evaluation step without mutating model state."""
    _, metrics = language_model_loss(
        model, batch, aux_loss_weight=aux_loss_weight
    )
    return metrics


def _aggregate_metrics(metrics: Iterable[dict[str, jax.Array]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    total_tokens = 0.0
    batches = 0
    for batch_metrics in metrics:
        token_count = float(batch_metrics["token_count"])
        total_tokens += token_count
        batches += 1
        for name in ("loss", "cross_entropy", "aux_loss", "accuracy"):
            totals[name] = totals.get(name, 0.0) + float(batch_metrics[name]) * token_count
    if batches == 0:
        raise ValueError("batch iterable cannot be empty")
    denominator = max(total_tokens, 1.0)
    result = {name: value / denominator for name, value in totals.items()}
    result["perplexity"] = float(np.exp(min(result["cross_entropy"], 20.0)))
    result["token_count"] = total_tokens
    return result


def train_epoch(
    model: KimiLinear,
    optimizer: nnx.Optimizer,
    batches: Iterable[LMBatch],
    config: TrainingConfig,
) -> dict[str, float]:
    """Train over an iterable of already-collated batches."""
    return _aggregate_metrics(
        train_step(
            model,
            optimizer,
            batch,
            aux_loss_weight=config.aux_loss_weight,
            router_bias_lr=config.router_bias_lr,
        )
        for batch in batches
    )


def evaluate(
    model: KimiLinear,
    batches: Iterable[LMBatch],
    config: TrainingConfig,
) -> dict[str, float]:
    """Evaluate over an iterable of already-collated batches."""
    return _aggregate_metrics(
        eval_step(model, batch, aux_loss_weight=config.aux_loss_weight)
        for batch in batches
    )


def save_checkpoint(
    directory: str | Path,
    model: KimiLinear,
    optimizer: nnx.Optimizer,
    *,
    model_config: KimiLinearConfig,
    training_config: TrainingConfig,
    force: bool = False,
) -> Path:
    """Save model/optimizer state plus versioned JSON configuration metadata."""
    path = Path(directory).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    state: dict[str, Any] = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
    }
    checkpointer.save(path, state, force=force)
    checkpointer.wait_until_finished()
    metadata = {
        "format_version": 1,
        "model_config": dataclasses.asdict(model_config),
        "training_config": dataclasses.asdict(training_config),
    }
    (path / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def restore_checkpoint(
    directory: str | Path,
    model: KimiLinear,
    optimizer: nnx.Optimizer,
) -> dict[str, Any]:
    """Restore into compatible model/optimizer objects and return metadata."""
    path = Path(directory).expanduser().resolve()
    metadata_path = path / "training_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError(
            f"unsupported checkpoint format version: {metadata.get('format_version')}"
        )

    target = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
    }
    restored = ocp.StandardCheckpointer().restore(path, target=target)
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    return metadata
