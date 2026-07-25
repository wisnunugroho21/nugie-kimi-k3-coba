"""NoPE Gated Multi-head Latent Attention (MLA) — the full-attention mixer.

In the Kimi Linear hybrid (Sec. 3 of the paper), 1 of every 4 layers is ordinary
softmax attention; this module is that layer, in Kimi Linear's exact flavor:

  * MLA (DeepSeek-V2 lineage): keys/values live in a small shared low-rank LATENT,
    so the decode-time cache stores one latent vector per position instead of full
    K and V — the whole point of MLA is that tiny KV cache.
  * NoPE — NO positional encoding of any kind. The GDN-2 linear layers already
    encode position implicitly through their recurrence, so Kimi Linear drops RoPE
    from its full-attention layers entirely (paper Sec. 3.3, "NoPE").
  * Written in the ABSORBED form (see the class docstring): with no RoPE in the
    way, the K/V up-projections fold into the neighboring matrices exactly, so the
    latent itself serves as both K and V and never gets up-projected at runtime.
  * Gated attention: a query-token-dependent sigmoid gate modulates every head
    channel after attention and before the output projection.  This is the Gated
    MLA layout used by Instella and the head-specific gating placement identified
    by "Gated Attention for Large Language Models."

Two paths, same math: `__call__` for full-sequence training (causal-masked matrix
attention) and `step` for streaming decode (append the new latent to a preallocated
cache, attend over it).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² =
# 2^{-5}), replacing Flax NNX's default Linear kernel init. Biases stay at zero.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")

F32 = jnp.float32


class MLACache(NamedTuple):
    """Streaming KV cache for the MLA layer. Thanks to MLA we cache only the small
    COMPRESSED latent `l_kv` (one latent serves as BOTH K and V — see below), in a
    preallocated [B, max_len, Hkv*Dh] buffer written at position `pos`. Unlike GDN-2's
    fixed-size state, this GROWS with context: these full-attention layers are exactly
    the ones that pay the long-context KV-cache cost in the hybrid (3:1 keeps them few)."""

    l_kv: jax.Array  # [B, max_len, num_kv_heads*head_dim]  preallocated latent buffer
    pos: jax.Array  # scalar int32: number of filled positions so far


class GroupedQueryLatentAttention(nnx.Module):
    """Grouped-query attention over a low-rank KV latent, with optional gating.

    This is NoPE (no rotary embeddings) Multi-head Latent Attention written in its
    matrix-absorbed form, fused with GQA-style KV-head sharing. Each of the three
    projections folds together two of the usual MLA matrices:

        w_q_uk : W_Q  . W_UK   -> queries are produced *directly* in the
                                  compressed K space, so they can dot against the
                                  latent without an explicit key up-projection.
        w_dkv  : W_DKV          -> down-projects x to the shared KV latent (c_kv).
        w_uv_o : W_UV . W_O     -> up-projects the value latent and applies the
                                  output projection in a single matmul.

    Key consequence: because there is no RoPE, W_UK and W_UV can be absorbed away
    *exactly*, and in the compressed latent space the keys and the values are the
    same tensor. That is why a single `l_kv` plays the role of BOTH K and V below.

    Note: `head_dim` here is the per-head latent (rank) dimension, not a
    conventional attention head width.
    """

    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rngs: nnx.Rngs,
        compute_dtype: jnp.dtype = F32,
        gated: bool = False,
        query_chunk_size: int = 128,
        cache_page_size: int = 64,
    ):
        # Matmul dtype for the projections (bf16 on H200); the QK^T / softmax / AV
        # core is upcast to fp32 below regardless, for a stable attention distribution.
        self.compute_dtype = compute_dtype
        # GQA constraint: every KV (latent) head must serve a whole number of
        # query heads.
        if (
            embed_dim < 1
            or num_q_heads < 1
            or num_kv_heads < 1
            or head_dim < 1
            or query_chunk_size < 1
            or cache_page_size < 1
        ):
            raise ValueError(
                "embed_dim, num_q_heads, num_kv_heads, head_dim, "
                "query_chunk_size, and cache_page_size must be positive"
            )
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gated = gated
        self.query_chunk_size = query_chunk_size
        self.cache_page_size = cache_page_size

        # How many query heads share each KV/latent head (the GQA group size).
        self.group_size = num_q_heads // num_kv_heads

        d_q = num_q_heads * head_dim  # total width of the query projection
        d_kv = num_kv_heads * head_dim  # total width of the (shared) KV latent

        # W_Q . W_UK absorbed: x -> queries already living in the latent K space.
        self.w_q_uk = nnx.Linear(
            embed_dim,
            d_q,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        # W_DKV: x -> low-rank KV latent c_kv (one latent per KV head).
        self.w_dkv = nnx.Linear(
            embed_dim,
            d_kv,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        # W_UV . W_O absorbed: value-latent -> up-projected, output-projected.
        self.w_uv_o = nnx.Linear(
            d_q,
            embed_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

        # Head-specific, element-wise sigmoid gate.  It is computed from the
        # current query token and therefore adds nothing to the KV cache.
        self.gate_proj = (
            nnx.Linear(
                embed_dim,
                d_q,
                use_bias=False,
                kernel_init=_XAVIER,
                dtype=compute_dtype,
                param_dtype=F32,
                rngs=rngs,
            )
            if gated
            else None
        )

    def _output(self, weighted_latents: jax.Array, x: jax.Array) -> jax.Array:
        """Apply the optional per-head gate before the absorbed output projection."""
        if self.gate_proj is not None:
            gate = jax.nn.sigmoid(self.gate_proj(x).astype(F32))
            weighted_latents = weighted_latents.astype(F32) * gate
        return self.w_uv_o(weighted_latents.astype(self.compute_dtype))

    def _group_queries(self, q_latent: jax.Array) -> jax.Array:
        """View query heads as [KV head, sharing group] without repeating KV."""
        batch_size, seq_length, _ = q_latent.shape
        return q_latent.reshape(
            batch_size,
            seq_length,
            self.num_kv_heads,
            self.group_size,
            self.head_dim,
        ).transpose(0, 2, 3, 1, 4)

    def _group_latents(self, l_kv: jax.Array) -> jax.Array:
        """Split the compressed latent into KV heads: [B, Hkv, T, Dh]."""
        batch_size, seq_length, _ = l_kv.shape
        return l_kv.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

    def _flatten_grouped(self, weighted: jax.Array) -> jax.Array:
        """Restore [B, T, Hq*Dh] from [B, Hkv, group, T, Dh]."""
        batch_size, _, _, seq_length, _ = weighted.shape
        return weighted.transpose(0, 3, 1, 2, 4).reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )

    def _attend_query_chunk(
        self,
        q: jax.Array,
        l_kv: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        """Grouped attention for one query chunk, with a fully safe softmax.

        q: [B, Hkv, group, Q, Dh], l_kv: [B, Hkv, K, Dh],
        mask: [B, Q, K]. The grouped layout broadcasts each KV head to its
        query-head group without allocating a repeated KV tensor.
        """
        logits = jnp.einsum("bhgqd,bhkd->bhgqk", q, l_kv).astype(F32) / jnp.sqrt(
            self.head_dim
        )
        expanded_mask = mask[:, None, None, :, :]
        masked_logits = jnp.where(expanded_mask, logits, -jnp.inf)
        has_keys = jnp.any(expanded_mask, axis=-1)
        row_max = jnp.max(masked_logits, axis=-1)
        safe_max = jnp.where(has_keys, row_max, 0.0)
        weights = jnp.where(
            expanded_mask,
            jnp.exp(logits - safe_max[..., None]),
            0.0,
        )
        denominator = jnp.maximum(jnp.sum(weights, axis=-1), 1.0)
        probabilities = (weights / denominator[..., None]).astype(l_kv.dtype)
        return jnp.einsum("bhgqk,bhkd->bhgqd", probabilities, l_kv)

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
    ) -> jax.Array:
        """Full-sequence causal attention.

        ``attention_mask`` is a boolean/numeric ``[B, T]`` mask. Masked positions
        are excluded as keys and produce an all-zero mixer delta as queries. The
        diagonal fallback for masked queries keeps softmax rows finite while the
        final output mask guarantees those rows cannot enter the residual stream.
        When supplied, integer ``segment_ids[B, T]`` also prevent attention across
        packed-document boundaries.
        """
        # x: (B, T, embed_dim)
        batch_size, seq_length, _ = x.shape
        if attention_mask is None:
            valid = jnp.ones((batch_size, seq_length), dtype=bool)
        else:
            if attention_mask.shape != (batch_size, seq_length):
                raise ValueError(
                    "attention_mask must have shape "
                    f"{(batch_size, seq_length)}, got {attention_mask.shape}"
                )
            valid = attention_mask.astype(bool)
        if segment_ids is not None:
            if segment_ids.shape != (batch_size, seq_length):
                raise ValueError(
                    "segment_ids must have shape "
                    f"{(batch_size, seq_length)}, got {segment_ids.shape}"
                )
            if not jnp.issubdtype(segment_ids.dtype, jnp.integer):
                raise TypeError("segment_ids must use an integer dtype")
        x = jnp.where(valid[..., None], x, 0)

        q_heads = self._group_queries(self.w_q_uk(x))
        l_kv_heads = self._group_latents(self.w_dkv(x))

        key_positions = jnp.arange(seq_length)
        if segment_ids is not None:
            previous_valid = jnp.pad(valid[:, :-1], ((0, 0), (1, 0)))
            previous_segment = jnp.pad(segment_ids[:, :-1], ((0, 0), (1, 0)))
            segment_start = valid & (
                (~previous_valid) | (segment_ids != previous_segment)
            )
            # Canonicalize IDs into contiguous runs. This prevents a reused raw
            # ID after a boundary from reconnecting to an earlier segment.
            segment_run = jnp.cumsum(segment_start, axis=1)
        else:
            segment_run = None

        def attend_chunk(
            query_chunk: jax.Array,
            query_valid: jax.Array,
            query_positions: jax.Array,
            query_segments: jax.Array | None,
        ) -> jax.Array:
            mask = (
                (key_positions[None, :] <= query_positions[:, None])[None, :, :]
                & query_valid[:, :, None]
                & valid[:, None, :]
            )
            if query_segments is not None:
                mask = mask & (query_segments[:, :, None] == segment_run[:, None, :])
            return self._attend_query_chunk(query_chunk, l_kv_heads, mask)

        chunk_size = min(self.query_chunk_size, seq_length)
        if seq_length <= chunk_size:
            weighted_heads = attend_chunk(q_heads, valid, key_positions, segment_run)
        else:
            num_chunks = (seq_length + chunk_size - 1) // chunk_size
            padded_length = num_chunks * chunk_size
            pad_length = padded_length - seq_length
            padded_q = jnp.pad(
                q_heads, ((0, 0), (0, 0), (0, 0), (0, pad_length), (0, 0))
            )
            padded_valid = jnp.pad(valid, ((0, 0), (0, pad_length)))
            padded_segments = (
                None
                if segment_run is None
                else jnp.pad(segment_run, ((0, 0), (0, pad_length)))
            )

            def compute_chunk(chunk_index: jax.Array) -> jax.Array:
                start = chunk_index * chunk_size
                query_chunk = jax.lax.dynamic_slice_in_dim(
                    padded_q, start, chunk_size, axis=3
                )
                query_valid = jax.lax.dynamic_slice_in_dim(
                    padded_valid, start, chunk_size, axis=1
                )
                query_segments = (
                    None
                    if padded_segments is None
                    else jax.lax.dynamic_slice_in_dim(
                        padded_segments, start, chunk_size, axis=1
                    )
                )
                return attend_chunk(
                    query_chunk,
                    query_valid,
                    start + jnp.arange(chunk_size),
                    query_segments,
                )

            # Rematerializing a chunk in backward keeps saved activation memory
            # proportional to one Q chunk rather than the full T-by-T score matrix.
            rematerialized_chunk = jax.checkpoint(compute_chunk)

            def scan_chunk(_, chunk_index):
                return None, rematerialized_chunk(chunk_index)

            _, chunks = jax.lax.scan(
                scan_chunk, None, jnp.arange(num_chunks, dtype=jnp.int32)
            )
            # [N, B, Hkv, G, C, Dh] -> [B, Hkv, G, N*C, Dh]
            weighted_heads = chunks.transpose(1, 2, 3, 0, 4, 5).reshape(
                batch_size,
                self.num_kv_heads,
                self.group_size,
                padded_length,
                self.head_dim,
            )[:, :, :, :seq_length, :]

        weighted_latents = self._flatten_grouped(weighted_heads)

        # Absorbed W_UV . W_O: up-project the value latent and output-project.
        output = self._output(weighted_latents, x)  # (B, T, embed_dim)

        return jnp.where(valid[..., None], output, 0)

    # ----------------------------------------------------------------------- #
    #  Streaming / inference.  Same softmax attention, but the KV latents of past
    #  positions are read from a preallocated cache instead of recomputed, and the
    #  new positions are written into it.  Use it for prefill (L = prompt length)
    #  and per-token decode (L = 1) alike.
    # ----------------------------------------------------------------------- #
    def init_cache(self, batch_size: int, max_len: int, dtype=None) -> MLACache:
        """Initialize the streaming KV cache for a given batch size and max length.
        The cache is a preallocated buffer of shape [B, max_len, Hkv*Dh] and a position counter.
        The buffer is filled with zeros initially."""
        if batch_size < 1 or max_len < 1:
            raise ValueError("batch_size and max_len must be positive")
        dtype = self.compute_dtype if dtype is None else dtype
        d_kv = self.num_kv_heads * self.head_dim
        return MLACache(
            l_kv=jnp.zeros((batch_size, max_len, d_kv), dtype),
            pos=jnp.array(0, jnp.int32),
        )

    def step(self, x: jax.Array, cache: MLACache) -> tuple[jax.Array, MLACache]:
        """Process a new chunk of input x, updating the cache and returning the output.
        x: [B, L, embed_dim]  cache: MLACache with l_kv: [B, max_len, Hkv*Dh], pos: scalar int32"""
        B, L, _ = x.shape
        max_len = cache.l_kv.shape[1]
        new_pos = cache.pos + L
        if L < 1:
            raise ValueError("Streaming MLA requires at least one input token")
        if L > max_len:
            raise ValueError(
                f"Input chunk length {L} exceeds MLA cache capacity {max_len}"
            )
        if cache.l_kv.shape[0] != B:
            raise ValueError(
                f"Cache batch size {cache.l_kv.shape[0]} does not match input {B}"
            )
        # Eager calls receive a concrete scalar and can report overflow cleanly.
        # Jitted decode is protected by KimiLinear.generate's static capacity check.
        if not isinstance(cache.pos, jax.core.Tracer):
            pos = int(cache.pos)
            if pos + L > max_len:
                raise ValueError(
                    f"MLA cache capacity {max_len} exceeded by positions "
                    f"[{pos}, {pos + L})"
                )

        # Queries for the new positions, grouped without repeating KV heads.
        q_heads = self._group_queries(self.w_q_uk(x))  # (B, Hkv, G, L, Dh)

        # New latents -> write them into the cache buffer at the current position.
        l_new = self.w_dkv(x)  # (B, L, Hkv*Dh)
        l_kv = jax.lax.dynamic_update_slice(
            cache.l_kv, l_new.astype(cache.l_kv.dtype), (0, cache.pos, 0)
        )

        # --- Shared KV latent (serves as both keys and values) ---
        l_kv_heads = self._group_latents(l_kv)

        # Page-wise online softmax never allocates scores over the full cache.
        # Empty pages bypass their dot products, so early decode scales with the
        # filled prefix rather than the declared maximum cache capacity.
        page_size = min(self.cache_page_size, max_len)
        num_pages = (max_len + page_size - 1) // page_size
        q_pos = cache.pos + jnp.arange(L)
        numerator = jnp.zeros(
            (B, self.num_kv_heads, self.group_size, L, self.head_dim), F32
        )
        denominator = jnp.zeros((B, self.num_kv_heads, self.group_size, L), F32)
        running_max = jnp.full_like(denominator, -jnp.inf)

        def page_body(page_index, state):
            current_max, current_denominator, current_numerator = state
            page_start = page_index * page_size

            def attend_page(state):
                current_max, current_denominator, current_numerator = state
                # dynamic_slice clamps at the end. The page_start lower bound in
                # page_mask excludes any overlap introduced for a short final page.
                slice_start = jnp.minimum(page_start, max_len - page_size)
                page = jax.lax.dynamic_slice_in_dim(
                    l_kv_heads, slice_start, page_size, axis=2
                )
                key_positions = slice_start + jnp.arange(page_size)
                page_end = jnp.minimum(page_start + page_size, new_pos)
                page_mask = (
                    (key_positions[None, :] >= page_start)
                    & (key_positions[None, :] < page_end)
                    & (key_positions[None, :] <= q_pos[:, None])
                )
                logits = jnp.einsum("bhgqd,bhkd->bhgqk", q_heads, page).astype(
                    F32
                ) / jnp.sqrt(self.head_dim)
                expanded_mask = page_mask[None, None, None, :, :]
                masked_logits = jnp.where(expanded_mask, logits, -jnp.inf)
                has_page = jnp.any(expanded_mask, axis=-1)
                page_max = jnp.max(masked_logits, axis=-1)
                safe_page_max = jnp.where(has_page, page_max, 0.0)
                page_weights = jnp.where(
                    expanded_mask,
                    jnp.exp(logits - safe_page_max[..., None]),
                    0.0,
                )
                page_denominator = jnp.sum(page_weights, axis=-1)
                page_numerator = jnp.einsum(
                    "bhgqk,bhkd->bhgqd", page_weights, page.astype(F32)
                )

                has_current = current_denominator > 0
                merged_max = jnp.where(
                    has_page,
                    jnp.where(
                        has_current,
                        jnp.maximum(current_max, page_max),
                        page_max,
                    ),
                    current_max,
                )
                merged_max = jax.lax.stop_gradient(merged_max)
                current_scale = jnp.where(
                    has_current, jnp.exp(current_max - merged_max), 0.0
                )
                page_scale = jnp.where(has_page, jnp.exp(page_max - merged_max), 0.0)
                merged_denominator = (
                    current_scale * current_denominator + page_scale * page_denominator
                )
                merged_numerator = (
                    current_scale[..., None] * current_numerator
                    + page_scale[..., None] * page_numerator
                )
                return merged_max, merged_denominator, merged_numerator

            return jax.lax.cond(
                page_start < new_pos, attend_page, lambda value: value, state
            )

        _, denominator, numerator = jax.lax.fori_loop(
            0,
            num_pages,
            page_body,
            (running_max, denominator, numerator),
        )
        weighted = numerator / jnp.maximum(denominator[..., None], 1.0)
        weighted = self._flatten_grouped(weighted.astype(l_kv.dtype))

        output = self._output(weighted, x)  # (B, L, embed_dim)
        return output, MLACache(l_kv, new_pos)


class GatedMultiHeadLatentAttention(GroupedQueryLatentAttention):
    """Explicit public name for MLA with the head-specific output gate enabled."""

    def __init__(self, *args, **kwargs):
        kwargs["gated"] = True
        super().__init__(*args, **kwargs)
