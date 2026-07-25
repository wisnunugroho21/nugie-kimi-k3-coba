"""Depth-wise Attention Residuals (AttnRes) for JAX / Flax NNX.

AttnRes treats preceding sub-layer outputs as values along the *depth* axis.
Every destination sub-layer owns one learned pseudo-query.  The corresponding
keys are RMS-normalized copies of the values:

    score_i = w^T RMSNorm(v_i)
    h       = sum_i softmax(score)_i v_i

The pseudo-query is deliberately initialized to zero, as prescribed by the
paper, so a newly initialized module starts as an equal-weight average.

For execution efficiency, parameter-free RMS-normalized sources are cached once.
Block AttnRes can batch every destination query over completed blocks (phase 1)
and merge the evolving partial block through online softmax (phase 2), avoiding
a full read and normalization of completed blocks at every sublayer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from gated_deltanet_2.layer import RMSNorm

F32 = jnp.float32


def _normalize_source(value: jax.Array, eps: float) -> jax.Array:
    """Parameter-free RMS normalization cached across destination queries."""
    value_f32 = value.astype(F32)
    inverse_rms = jax.lax.rsqrt(
        jnp.mean(value_f32 * value_f32, axis=-1, keepdims=True) + eps
    )
    # Match RMSNorm's mixed-precision boundary: normalize in fp32, then return
    # to the source dtype before the destination-specific scale is applied.
    return (value_f32 * inverse_rms).astype(value.dtype)


class AttentionResidualState(NamedTuple):
    """Depth sources plus reusable parameter-free normalized keys."""

    values: jax.Array  # [S, B, T, D]
    normalized: jax.Array  # [S, B, T, D]

    @classmethod
    def initialize(cls, value: jax.Array, *, eps: float) -> "AttentionResidualState":
        if value.ndim != 3:
            raise ValueError(
                f"AttnRes source must have shape [batch, length, d_model], got {value.shape}"
            )
        return cls(value[None], _normalize_source(value, eps)[None])

    def append(self, value: jax.Array, *, eps: float) -> "AttentionResidualState":
        if value.shape != self.values.shape[1:]:
            raise ValueError(
                f"AttnRes source must have shape {self.values.shape[1:]}, got {value.shape}"
            )
        return AttentionResidualState(
            jnp.concatenate((self.values, value[None]), axis=0),
            jnp.concatenate(
                (self.normalized, _normalize_source(value, eps)[None]), axis=0
            ),
        )


class AttentionResidualPhase(NamedTuple):
    """Unnormalized inter-block outputs and online-softmax statistics."""

    numerator: jax.Array  # [Q, B, T, D]
    max_score: jax.Array  # [Q, B, T]
    denominator: jax.Array  # [Q, B, T]
    output_dtype: jnp.dtype


class AttentionResidual(nnx.Module):
    """Single-query softmax attention over a sequence of depth-wise values."""

    def __init__(self, d_model: int, *, eps: float = 1e-5, rngs: nnx.Rngs):
        self.key_norm = RMSNorm(d_model, eps=eps, rngs=rngs)
        self.query = nnx.Param(jnp.zeros((d_model,), F32))

    def __call__(
        self,
        values: Sequence[jax.Array] | jax.Array | AttentionResidualState,
        *,
        partial: jax.Array | None = None,
        return_weights: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Aggregate depth sources.

        Args:
            values: A prepared source cache, a non-empty sequence of ``[B, L, D]``
                tensors, or an already stacked ``[S, B, L, D]`` tensor.
            partial: Optional evolving intra-block source ``[B, L, D]``.
            return_weights: Also return ``[S, B, L]`` depth-attention weights.
        """
        if isinstance(values, AttentionResidualState):
            stacked = values.values
            normalized = values.normalized
        elif isinstance(values, jax.Array):
            stacked = values
            normalized = _normalize_source(stacked, self.key_norm.eps)
        else:
            if not values:
                raise ValueError("AttentionResidual requires at least one source")
            stacked = jnp.stack(tuple(values), axis=0)
            normalized = _normalize_source(stacked, self.key_norm.eps)

        if stacked.ndim != 4:
            raise ValueError(
                "AttnRes values must have shape [sources, batch, length, d_model], "
                f"got {stacked.shape}"
            )
        if partial is not None:
            if partial.shape != stacked.shape[1:]:
                raise ValueError(
                    f"AttnRes partial must have shape {stacked.shape[1:]}, "
                    f"got {partial.shape}"
                )
            stacked = jnp.concatenate((stacked, partial[None]), axis=0)
            normalized = jnp.concatenate(
                (
                    normalized,
                    _normalize_source(partial, self.key_norm.eps)[None],
                ),
                axis=0,
            )

        effective_query = (
            self.query[...].astype(F32)
            * self.key_norm.weight[...].astype(F32)
        )
        logits = jnp.einsum(
            "d,sbtd->sbt", effective_query, normalized.astype(F32)
        )
        weights = jax.nn.softmax(logits, axis=0)
        output = jnp.einsum("sbt,sbtd->btd", weights.astype(stacked.dtype), stacked)
        if return_weights:
            return output, weights
        return output

    def merge_phase(
        self,
        phase: AttentionResidualPhase,
        query_index: int,
        partial: jax.Array | None,
    ) -> jax.Array:
        """Merge cached inter-block attention with one evolving partial source."""
        numerator = phase.numerator[query_index]
        maximum = phase.max_score[query_index]
        denominator = phase.denominator[query_index]
        if partial is None:
            return (numerator / denominator[..., None]).astype(phase.output_dtype)

        effective_query = (
            self.query[...].astype(F32)
            * self.key_norm.weight[...].astype(F32)
        )
        partial_key = _normalize_source(partial, self.key_norm.eps).astype(F32)
        partial_score = jnp.einsum("d,btd->bt", effective_query, partial_key)

        merged_max = jax.lax.stop_gradient(
            jnp.maximum(maximum, partial_score)
        )
        inter_scale = jnp.exp(maximum - merged_max)
        partial_scale = jnp.exp(partial_score - merged_max)
        merged_denominator = inter_scale * denominator + partial_scale
        merged_numerator = (
            inter_scale[..., None] * numerator
            + partial_scale[..., None] * partial.astype(F32)
        )
        return (merged_numerator / merged_denominator[..., None]).astype(
            phase.output_dtype
        )


def prepare_batched_attention_residual(
    residuals: Sequence[AttentionResidual],
    sources: AttentionResidualState,
) -> AttentionResidualPhase:
    """Phase 1: batch a block's pseudo-queries over completed depth sources."""
    if not residuals:
        raise ValueError("At least one AttentionResidual query is required")
    effective_queries = jnp.stack(
        tuple(
            residual.query[...].astype(F32)
            * residual.key_norm.weight[...].astype(F32)
            for residual in residuals
        ),
        axis=0,
    )
    logits = jnp.einsum(
        "qd,sbtd->qsbt", effective_queries, sources.normalized.astype(F32)
    )
    maximum = jax.lax.stop_gradient(jnp.max(logits, axis=1))
    unnormalized = jnp.exp(logits - maximum[:, None])
    denominator = jnp.sum(unnormalized, axis=1)
    numerator = jnp.einsum(
        "qsbt,sbtd->qbtd", unnormalized, sources.values.astype(F32)
    )
    return AttentionResidualPhase(
        numerator=numerator,
        max_score=maximum,
        denominator=denominator,
        output_dtype=sources.values.dtype,
    )
