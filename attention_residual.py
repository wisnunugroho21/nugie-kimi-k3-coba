"""Depth-wise Attention Residuals (AttnRes) for JAX / Flax NNX.

AttnRes treats preceding sub-layer outputs as values along the *depth* axis.
Every destination sub-layer owns one learned pseudo-query.  The corresponding
keys are RMS-normalized copies of the values:

    score_i = w^T RMSNorm(v_i)
    h       = sum_i softmax(score)_i v_i

The pseudo-query is deliberately initialized to zero, as prescribed by the
paper, so a newly initialized module starts as an equal-weight average.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from gated_deltanet_2.layer import RMSNorm

F32 = jnp.float32


class AttentionResidual(nnx.Module):
    """Single-query softmax attention over a sequence of depth-wise values."""

    def __init__(self, d_model: int, *, eps: float = 1e-5, rngs: nnx.Rngs):
        self.key_norm = RMSNorm(d_model, eps=eps, rngs=rngs)
        self.query = nnx.Param(jnp.zeros((d_model,), F32))

    def __call__(
        self,
        values: Sequence[jax.Array] | jax.Array,
        *,
        return_weights: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Aggregate depth sources.

        Args:
            values: A non-empty sequence of ``[B, L, D]`` tensors, or an already
                stacked tensor with shape ``[S, B, L, D]``.
            return_weights: Also return ``[S, B, L]`` depth-attention weights.
        """
        if isinstance(values, jax.Array):
            stacked = values
        else:
            if not values:
                raise ValueError("AttentionResidual requires at least one source")
            stacked = jnp.stack(tuple(values), axis=0)

        if stacked.ndim != 4:
            raise ValueError(
                "AttnRes values must have shape [sources, batch, length, d_model], "
                f"got {stacked.shape}"
            )

        keys = self.key_norm(stacked).astype(F32)
        logits = jnp.einsum("d,sbtd->sbt", self.query[...].astype(F32), keys)
        weights = jax.nn.softmax(logits, axis=0)
        output = jnp.einsum("sbt,sbtd->btd", weights.astype(stacked.dtype), stacked)
        if return_weights:
            return output, weights
        return output
