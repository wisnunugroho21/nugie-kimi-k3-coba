"""Minimal Kimi K3 text backbone in JAX and Flax NNX.

The implementation follows equations (1), (2), (5)-(14) of *Kimi K3: Open
Frontier Intelligence* where doing so remains understandable. It deliberately
uses ordinary JAX operations instead of fused KDA, FlashAttention, expert
parallelism, or custom quantized kernels.

Tensor notation used throughout:

* ``B`` - batch size
* ``T`` - sequence length
* ``D`` - model width
* ``H`` - attention heads
* ``P`` - per-head width
* ``E`` - routed experts
* ``L`` - latent MoE width
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from kimi_k3.config import KimiK3Config

Array = jax.Array


def _normal_init(stddev: float):
    """Return the normal initializer used by the released model."""
    return jax.nn.initializers.normal(stddev)


def _linear(
    in_features: int,
    out_features: int,
    *,
    config: KimiK3Config,
    rngs: nnx.Rngs,
) -> nnx.Linear:
    """Create a bias-free projection, as used throughout the K3 backbone."""
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=False,
        kernel_init=_normal_init(config.initializer_std),
        rngs=rngs,
    )


def softcap(x: Array, cap: float) -> Array:
    """Smoothly limit ``x`` to ``[-cap, cap]`` with ``cap * tanh(x/cap)``."""
    return cap * jnp.tanh(x / cap)


class SiTUGLU(nnx.Module):
    """Sigmoid Tanh Unit GLU from paper equation (12).

    The gate branch resembles Swish near zero, but both multiplicative branches
    are bounded. With K3's beta values, each scalar product is bounded by
    ``4 * 25 = 100``, which reduces low-precision activation explosions.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        config: KimiK3Config,
        rngs: nnx.Rngs,
    ):
        self.gate_proj = _linear(
            in_features, hidden_features, config=config, rngs=rngs
        )
        self.up_proj = _linear(
            in_features, hidden_features, config=config, rngs=rngs
        )
        self.down_proj = _linear(
            hidden_features, out_features, config=config, rngs=rngs
        )
        self.gate_beta = config.situ_gate_beta
        self.up_beta = config.situ_up_beta

    def activated_hidden(self, x: Array) -> Array:
        """Return the bounded hidden activation before its output projection."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        bounded_swish = softcap(gate, self.gate_beta) * jax.nn.sigmoid(gate)
        bounded_up = softcap(up, self.up_beta)
        return bounded_swish * bounded_up

    def __call__(self, x: Array) -> Array:
        """Apply the two gated branches followed by the down projection."""
        return self.down_proj(self.activated_hidden(x))


class CausalDepthwiseConv1D(nnx.Module):
    """A transparent replacement for K3's fused ShortConv operation.

    Every channel has its own small causal kernel. Left padding ensures that
    output position ``t`` only sees positions up to and including ``t``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        config: KimiK3Config,
        rngs: nnx.Rngs,
    ):
        self.channels = channels
        self.kernel_size = kernel_size
        key = rngs.params()
        values = _normal_init(config.initializer_std)(
            key, (kernel_size, 1, channels), jnp.float32
        )
        self.kernel = nnx.Param(values)

    def __call__(self, x: Array) -> Array:
        """Convolve an array shaped ``[B, T, channels]`` causally."""
        return jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.kernel[...],
            window_strides=(1,),
            padding=((self.kernel_size - 1, 0),),
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=self.channels,
        )


class KimiDeltaAttention(nnx.Module):
    """Readable recurrent Kimi Delta Attention (KDA).

    This is equation (1) evaluated one token at a time with ``jax.lax.scan``.
    The production implementation uses the algebraically equivalent chunkwise
    algorithm in equation (4), which is much faster but substantially harder to
    study. The lower-bounded decay from equation (5), ShortConv preprocessing,
    L2-normalized queries/keys, and full-rank output gate are retained.
    """

    def __init__(self, config: KimiK3Config, *, rngs: nnx.Rngs):
        self.config = config
        projection_size = config.num_heads * config.head_dim

        self.q_proj = _linear(
            config.hidden_size, projection_size, config=config, rngs=rngs
        )
        self.k_proj = _linear(
            config.hidden_size, projection_size, config=config, rngs=rngs
        )
        self.v_proj = _linear(
            config.hidden_size, projection_size, config=config, rngs=rngs
        )
        self.q_conv = CausalDepthwiseConv1D(
            projection_size,
            config.short_conv_kernel_size,
            config=config,
            rngs=rngs,
        )
        self.k_conv = CausalDepthwiseConv1D(
            projection_size,
            config.short_conv_kernel_size,
            config=config,
            rngs=rngs,
        )
        self.v_conv = CausalDepthwiseConv1D(
            projection_size,
            config.short_conv_kernel_size,
            config=config,
            rngs=rngs,
        )

        # W_alpha_down and W_alpha_up implement the low-rank decay logits in
        # equation (2). The head-wise scale A starts at zero as stated in §2.1.1.
        self.decay_down = _linear(
            config.hidden_size, config.head_dim, config=config, rngs=rngs
        )
        self.decay_up = _linear(
            config.head_dim, projection_size, config=config, rngs=rngs
        )
        self.decay_bias = nnx.Param(jnp.zeros((config.num_heads, config.head_dim)))
        self.log_decay_scale = nnx.Param(jnp.zeros((config.num_heads, 1)))

        self.beta_proj = _linear(
            config.hidden_size, config.num_heads, config=config, rngs=rngs
        )
        self.output_gate = _linear(
            config.hidden_size, projection_size, config=config, rngs=rngs
        )
        self.output_norm = nnx.RMSNorm(
            config.head_dim, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.output_proj = _linear(
            projection_size, config.hidden_size, config=config, rngs=rngs
        )

    def _split_heads(self, x: Array) -> Array:
        """Convert ``[B, T, H*P]`` to ``[B, T, H, P]``."""
        return x.reshape(
            x.shape[0], x.shape[1], self.config.num_heads, self.config.head_dim
        )

    def retention(self, x: Array) -> Array:
        """Compute channel-wise retention ``alpha`` from paper equation (5)."""
        z = self._split_heads(self.decay_up(self.decay_down(x)))
        z = z + self.decay_bias[...]
        scale = jnp.exp(self.log_decay_scale[...])
        log_decay = self.config.gate_lower_bound * jax.nn.sigmoid(scale * z)
        return jnp.exp(log_decay)

    def __call__(
        self,
        x: Array,
        *,
        initial_state: Array | None = None,
    ) -> tuple[Array, Array]:
        """Mix a full sequence and return ``(outputs, final_recurrent_state)``.

        ``initial_state`` has shape ``[B, H, P, P]``. Passing it is useful when
        demonstrating streaming; the miniature backbone itself processes full
        sequences and starts from zeros.
        """
        batch_size = x.shape[0]
        q = self._split_heads(jax.nn.silu(self.q_conv(self.q_proj(x))))
        k = self._split_heads(jax.nn.silu(self.k_conv(self.k_proj(x))))
        v = self._split_heads(jax.nn.silu(self.v_conv(self.v_proj(x))))
        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-6)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)

        alpha = self.retention(x)
        beta = jax.nn.sigmoid(self.beta_proj(x))
        if initial_state is None:
            initial_state = jnp.zeros(
                (
                    batch_size,
                    self.config.num_heads,
                    self.config.head_dim,
                    self.config.head_dim,
                ),
                dtype=x.dtype,
            )

        # lax.scan iterates over time, so move T to the leading dimension.
        scan_inputs = tuple(
            value.swapaxes(0, 1) for value in (q, k, v, alpha, beta)
        )

        def recurrent_step(
            state: Array,
            inputs: tuple[Array, Array, Array, Array, Array],
        ) -> tuple[Array, Array]:
            q_t, k_t, v_t, alpha_t, beta_t = inputs

            # Eq. (1), rearranged into a delta-rule form:
            # retained = Diag(alpha) @ S_(t-1)
            # S_t = retained + beta*k*(v - k^T retained)^T
            retained = alpha_t[..., :, None] * state
            prediction = jnp.einsum("bhp,bhpv->bhv", k_t, retained)
            error = v_t - prediction
            state = retained + (
                beta_t[..., None, None]
                * k_t[..., :, None]
                * error[..., None, :]
            )
            output = jnp.einsum("bhpv,bhp->bhv", state, q_t)
            return state, output

        final_state, output = jax.lax.scan(
            recurrent_step, initial_state, scan_inputs
        )
        output = output.swapaxes(0, 1)  # [T,B,H,P] -> [B,T,H,P]

        # Equation (6): head-wise RMSNorm, then a data-dependent full-rank gate.
        output = self.output_norm(output)
        gate = jax.nn.sigmoid(self._split_heads(self.output_gate(x)))
        output = output * gate
        output = output.reshape(batch_size, x.shape[1], -1)
        return self.output_proj(output), final_state


class GatedMLA(nnx.Module):
    """NoPE Multi-head Latent Attention with K3's output gate.

    MLA stores each token's key/value information in a small latent vector and
    reconstructs head-specific keys and values only for attention. This example
    evaluates ordinary quadratic causal attention; a production decoder would
    cache the latent vector instead of the expanded K/V tensors.
    """

    def __init__(self, config: KimiK3Config, *, rngs: nnx.Rngs):
        self.config = config
        hp = config.num_heads * config.head_dim
        self.q_down = _linear(
            config.hidden_size, config.q_lora_rank, config=config, rngs=rngs
        )
        self.q_norm = nnx.RMSNorm(
            config.q_lora_rank, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.q_up = _linear(
            config.q_lora_rank, hp, config=config, rngs=rngs
        )

        self.kv_down = _linear(
            config.hidden_size, config.kv_lora_rank, config=config, rngs=rngs
        )
        self.kv_norm = nnx.RMSNorm(
            config.kv_lora_rank, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.kv_up = _linear(
            config.kv_lora_rank, 2 * hp, config=config, rngs=rngs
        )
        self.output_gate = _linear(
            config.hidden_size, hp, config=config, rngs=rngs
        )
        self.output_proj = _linear(
            hp, config.hidden_size, config=config, rngs=rngs
        )

    def _heads(self, x: Array) -> Array:
        """Convert a packed head dimension to ``[B, T, H, P]``."""
        return x.reshape(
            x.shape[0], x.shape[1], self.config.num_heads, self.config.head_dim
        )

    def __call__(self, x: Array) -> Array:
        """Apply causal global attention without positional encoding."""
        q = self._heads(self.q_up(self.q_norm(self.q_down(x))))
        compressed_kv = self.kv_norm(self.kv_down(x))
        k, v = jnp.split(self.kv_up(compressed_kv), 2, axis=-1)
        k, v = self._heads(k), self._heads(v)

        # Scores are [B,H,query_position,key_position]. K3 keeps global
        # attention outputs in FP32 during training to reduce rounding bias.
        scores = jnp.einsum("bthp,bshp->bhts", q, k).astype(jnp.float32)
        scores = scores / math.sqrt(self.config.head_dim)
        causal = jnp.tril(jnp.ones((x.shape[1], x.shape[1]), dtype=jnp.bool_))
        scores = jnp.where(causal[None, None, :, :], scores, -jnp.inf)
        probabilities = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum(
            "bhts,bshp->bthp", probabilities, v.astype(jnp.float32)
        )

        # Equation (7): an input-dependent channel-wise full-rank output gate.
        gate = jax.nn.sigmoid(self._heads(self.output_gate(x)))
        output = output.astype(x.dtype) * gate
        return self.output_proj(output.reshape(x.shape[0], x.shape[1], -1))


class RouterBias(nnx.Variable):
    """Non-gradient state updated by Quantile Balancing, not by backprop."""


class QuantileRouter(nnx.Module):
    """Sigmoid Top-k router and exact-batch Quantile Balancing.

    The paper uses histograms to estimate global quantiles across a distributed
    batch. Here ``update_bias`` computes the exact quantile for a small local
    batch, which expresses equation (14) more directly.
    """

    def __init__(self, config: KimiK3Config, *, rngs: nnx.Rngs):
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.score = _linear(
            config.hidden_size, config.num_experts, config=config, rngs=rngs
        )
        self.correction_bias = RouterBias(jnp.zeros((config.num_experts,)))

    def __call__(self, x: Array) -> tuple[Array, Array, Array]:
        """Return ``(expert_indices, mixture_weights, raw_sigmoid_scores)``."""
        raw_scores = jax.nn.sigmoid(self.score(x).astype(jnp.float32))

        # The correction bias affects expert selection only. Per equation (13),
        # the actual mixture weights come from the raw, unbiased scores.
        selection_scores = raw_scores + self.correction_bias[...]
        _, expert_indices = jax.lax.top_k(selection_scores, self.top_k)
        weights = jnp.take_along_axis(raw_scores, expert_indices, axis=-1)
        weights = weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
        return expert_indices, weights.astype(x.dtype), raw_scores

    def update_bias(self, raw_scores: Array) -> Array:
        """Apply exact Quantile Balancing and return the new centered bias.

        This is a state update intended to run *after* routing a training batch;
        the new value therefore affects only the next batch, preserving the
        causality requirement in §2.3.3.
        """
        scores = raw_scores.reshape(-1, self.num_experts).astype(jnp.float32)
        biased_scores = scores + self.correction_bias[...]
        top_k_plus_one, _ = jax.lax.top_k(biased_scores, self.top_k + 1)
        cutoff = top_k_plus_one[:, -1]

        # For expert j, choose the bias whose threshold admits a k/E fraction
        # of tokens. A shared offset is removed because it cannot affect Top-k.
        margins = scores - cutoff[:, None]
        quantile = jnp.quantile(
            margins, 1.0 - self.top_k / self.num_experts, axis=0
        )
        new_bias = -quantile
        new_bias = new_bias - new_bias.mean()
        self.correction_bias[...] = new_bias
        return new_bias


class StableLatentMoE(nnx.Module):
    """Stable LatentMoE from paper equation (11).

    Routed experts work at width ``L < D`` while shared experts keep the full
    model width. For clarity this implementation evaluates every routed expert
    and masks unselected outputs. Real K3 dispatches tokens only to their 16
    selected experts across an expert-parallel cluster.
    """

    def __init__(self, config: KimiK3Config, *, rngs: nnx.Rngs):
        self.config = config
        self.router = QuantileRouter(config, rngs=rngs)
        self.latent_down = _linear(
            config.hidden_size, config.latent_moe_dim, config=config, rngs=rngs
        )
        # nnx.List registers every expert as a nested NNX module, so parameters
        # remain visible to transformations such as nnx.grad and nnx.state.
        self.experts = nnx.List(
            [
                SiTUGLU(
                    config.latent_moe_dim,
                    config.moe_hidden_size,
                    config.latent_moe_dim,
                    config=config,
                    rngs=rngs,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.routed_norm = nnx.RMSNorm(
            config.latent_moe_dim, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.latent_up = _linear(
            config.latent_moe_dim, config.hidden_size, config=config, rngs=rngs
        )

        # The official inference code represents N shared experts as one GLU
        # with N times the hidden width; this is equivalent to concatenating N
        # independently parameterized expert branches before the down map.
        self.shared_experts = SiTUGLU(
            config.hidden_size,
            config.num_shared_experts * config.moe_hidden_size,
            config.hidden_size,
            config=config,
            rngs=rngs,
        )

    def __call__(
        self, x: Array, *, return_router_scores: bool = False
    ) -> Array | tuple[Array, Array]:
        """Mix selected routed experts with the full-width shared path."""
        expert_indices, weights, raw_scores = self.router(x)
        latent = self.latent_down(x)

        # Convert K selected indices and weights into one sparse E-vector.
        selected = jax.nn.one_hot(
            expert_indices, self.config.num_experts, dtype=x.dtype
        )
        mixture = (selected * weights[..., None]).sum(axis=-2)

        # [B,T,E,L]. This dense pedagogical form is easy to inspect and
        # differentiate, but should be replaced by token dispatch at scale.
        all_expert_outputs = jnp.stack(
            [expert(latent) for expert in self.experts], axis=-2
        )
        routed = jnp.einsum("bte,btel->btl", mixture, all_expert_outputs)
        routed = self.latent_up(self.routed_norm(routed))
        output = self.shared_experts(x) + routed
        if return_router_scores:
            return output, raw_scores
        return output


class AttentionResidual(nnx.Module):
    """Attend over depth using a learned pseudo-query (equations 8-10)."""

    def __init__(self, config: KimiK3Config, *, rngs: nnx.Rngs):
        self.source_norm = nnx.RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.pseudo_query = _linear(
            config.hidden_size, 1, config=config, rngs=rngs
        )

    def __call__(self, current: Array, completed_blocks: Sequence[Array]) -> Array:
        """Retrieve from completed blocks plus the current block's partial sum."""
        sources = jnp.stack([*completed_blocks, current], axis=-2)
        scores = self.pseudo_query(self.source_norm(sources)).squeeze(-1)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        return jnp.einsum(
            "bts,btsd->btd", weights, sources.astype(jnp.float32)
        ).astype(current.dtype)


class DecoderLayer(nnx.Module):
    """One K3 attention mixer followed by dense or sparse channel mixing."""

    def __init__(
        self, config: KimiK3Config, layer_index: int, *, rngs: nnx.Rngs
    ):
        self.layer_index = layer_index
        self.block_size = config.attn_res_block_size
        self.attention_kind = config.attention_kind(layer_index)
        if self.attention_kind == "kda":
            self.attention = KimiDeltaAttention(config, rngs=rngs)
        else:
            self.attention = GatedMLA(config, rngs=rngs)

        if layer_index < config.first_dense_layers:
            self.feed_forward = SiTUGLU(
                config.hidden_size,
                config.dense_hidden_size,
                config.hidden_size,
                config=config,
                rngs=rngs,
            )
        else:
            self.feed_forward = StableLatentMoE(config, rngs=rngs)

        self.pre_attention_residual = AttentionResidual(config, rngs=rngs)
        self.pre_moe_residual = AttentionResidual(config, rngs=rngs)
        self.attention_norm = nnx.RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.moe_norm = nnx.RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )

    def __call__(
        self, hidden: Array, completed_blocks: list[Array]
    ) -> tuple[Array, list[Array]]:
        """Run the layer while maintaining Block Attention Residual state."""
        prefix_sum: Array | None = hidden

        # At the first layer of each block, cache the just-completed prefix.
        # Later layers retrieve from those cached block summaries and the
        # current block's running sum.
        if completed_blocks:
            attention_input = self.pre_attention_residual(
                prefix_sum, completed_blocks
            )
        else:
            attention_input = prefix_sum

        if self.layer_index % self.block_size == 0:
            completed_blocks = [*completed_blocks, prefix_sum]
            prefix_sum = None

        attention_input = self.attention_norm(attention_input)
        if self.attention_kind == "kda":
            attention_output, _ = self.attention(attention_input)
        else:
            attention_output = self.attention(attention_input)
        prefix_sum = (
            attention_output
            if prefix_sum is None
            else prefix_sum + attention_output
        )

        moe_input = self.pre_moe_residual(prefix_sum, completed_blocks)
        moe_output = self.feed_forward(self.moe_norm(moe_input))
        return prefix_sum + moe_output, completed_blocks


class KimiK3(nnx.Module):
    """A small decoder-only Kimi K3 text model.

    Input token IDs have shape ``[B, T]`` and output logits have shape
    ``[B, T, vocab_size]``. This class intentionally excludes MoonViT-V2; visual
    features in the real model are projected into the same embedding stream
    before entering this backbone.
    """

    def __init__(self, config: KimiK3Config, *, rngs: nnx.Rngs):
        self.config = config
        self.token_embedding = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=_normal_init(config.initializer_std),
            rngs=rngs,
        )
        self.layers = nnx.List(
            [
                DecoderLayer(config, index, rngs=rngs)
                for index in range(config.num_layers)
            ]
        )
        self.output_residual = AttentionResidual(config, rngs=rngs)
        self.final_norm = nnx.RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs
        )
        self.lm_head = _linear(
            config.hidden_size, config.vocab_size, config=config, rngs=rngs
        )

    def __call__(
        self, token_ids: Array, *, return_hidden: bool = False
    ) -> Array | tuple[Array, Array]:
        """Return next-token logits, optionally together with final features."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {token_ids.shape[1]} exceeds configured "
                f"maximum {self.config.max_sequence_length}"
            )

        hidden = self.token_embedding(token_ids)
        completed_blocks: list[Array] = []
        for layer in self.layers:
            hidden, completed_blocks = layer(hidden, completed_blocks)

        # The paper's final output layer also retrieves all block summaries.
        hidden = self.output_residual(hidden, completed_blocks)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        if return_hidden:
            return logits, hidden
        return logits


def causal_lm_loss(
    logits: Array,
    token_ids: Array,
    *,
    token_mask: Array | None = None,
) -> Array:
    """Compute mean next-token cross entropy without an Optax dependency.

    ``token_mask`` may have the same shape as ``token_ids``; zero entries omit
    target positions from the average. Position zero is never a target because
    its token is used to predict position one.
    """
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token_ids must share batch/sequence shapes")
    predicted = logits[:, :-1].astype(jnp.float32)
    targets = token_ids[:, 1:]
    target_log_probs = jnp.take_along_axis(
        jax.nn.log_softmax(predicted, axis=-1), targets[..., None], axis=-1
    ).squeeze(-1)

    if token_mask is None:
        return -target_log_probs.mean()
    weights = token_mask[:, 1:].astype(jnp.float32)
    return -(target_log_probs * weights).sum() / jnp.maximum(weights.sum(), 1.0)
