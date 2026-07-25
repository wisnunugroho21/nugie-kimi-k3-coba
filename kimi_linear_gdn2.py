"""
Kimi Linear (GDN-2 variant) — the top-level decoder-only language model, in JAX /
Flax NNX. ANNOTATED against "Kimi Linear: An Expressive, Efficient Attention
Architecture."

WHAT KIMI LINEAR IS (paper, Sec. 3 / Fig. 2)
--------------------------------------------
A *hybrid* linear-attention transformer. Most layers use a cheap, O(L) linear-
attention token mixer (the paper's "Kimi Delta Attention", KDA); a minority use
ordinary softmax full attention (Multi-head Latent Attention, MLA). The two are
interleaved at a fixed **3:1 ratio** — three linear layers for every one full-
attention layer — which the paper finds recovers full-attention quality at a
fraction of the KV-cache and compute cost.

  • KDA layers carry positional information implicitly through their recurrence,
    so the full-attention layers need NO positional encoding. Hence the MLA layers
    here are NoPE (see multi_latent_attention/attention.py).
  • Every layer's channel mixer is a LatentMoE: routed experts work in a compressed
    space while routing and the shared expert stay at model width.
  • Residual streams use Block Attention Residuals (AttnRes), selecting earlier
    block representations with token-dependent softmax weights over depth.
  • Full-attention layers use a per-head sigmoid output gate (Gated MLA).

THIS FILE'S ONE DELIBERATE SUBSTITUTION
---------------------------------------
We replace KDA with **Gated DeltaNet-2** ("Decoupling Erase and Write in Linear
Attention", arXiv:2605.22791). Both are gated-delta-rule linear attentions with
fine-grained (channel-wise) gating; GDN-2's twist is a separate erase gate `b` and
write gate `w` instead of the single `beta` that KDA/GDN share. The 3:1 hybrid
schedule and NoPE MLA layout are retained, then extended below with AttnRes,
LatentMoE, and gated MLA. See gated_deltanet_2/layer.py for the linear token mixer.

BLOCK STRUCTURE
---------------
    h = AttnRes(depth_sources)
    token_delta = TokenMixer(RMSNorm(h))
    h = AttnRes(depth_sources + token_delta)
    mlp_delta = LatentMoE(RMSNorm(h))

MODEL = Embed -> [DecoderLayer] * n_layers -> RMSNorm -> LM head.

TWO FORWARD MODES
-----------------
  • Training / full sequence:  model(input_ids)  — parallel, GDN-2 via its chunkwise
    core, MLA via a full causal-attention matrix.
  • Streaming / inference:     model.step(ids, caches) and model.generate(...)  —
    reuses per-layer state across calls so each new token is O(1) work for the GDN-2
    layers (fixed-size recurrent state) and O(context) for the few MLA layers (growing
    latent cache). See GatedDeltaNet2.step / GroupedQueryLatentAttention.step.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from attention_residual import AttentionResidual

# Reuse the building blocks already implemented and verified in this repo.
from gated_deltanet_2.layer import GatedDeltaNet2, GDN2Cache, RMSNorm
from multi_latent_attention.attention import (
    GatedMultiHeadLatentAttention,
    GroupedQueryLatentAttention,
    MLACache,
)
from multi_latent_attention.moe import GroupedGemmMoE, LatentMoE

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² =
# 2^{-5}) for the embedding and LM head, replacing Flax NNX's defaults. The (small)
# embedding scale this produces is fine — RMSNorm renormalizes the residual stream.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


# --------------------------------------------------------------------------- #
#  Configuration
#
#  Defaults are deliberately TINY so the whole model trains on a laptop CPU. The
#  paper's 48B-A3B numbers are quoted in comments for reference; only the *ratios*
#  and structure matter for understanding — scale up by raising the dims/layers.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class KimiLinearConfig:
    vocab_size: int = 256  # paper: 160k; tiny here (byte-level demo)
    d_model: int = 256  # model width  (paper 1.3B: 2048)
    n_layers: int = 8  # depth        (paper 1.3B: 27)

    # --- Hybrid schedule: which layers are FULL attention (MLA) vs linear (GDN-2) ---
    # full_attn_period = 4 places one MLA layer every 4th layer (indices 3, 7, ...),
    # i.e. a 3:1 linear:full ratio — exactly Kimi Linear's hybrid recipe (Sec. 3.2).
    full_attn_period: int = 4

    # --- GDN-2 token mixer (the KDA replacement) — see gated_deltanet_2/layer.py ---
    gdn_num_heads: int = 4  # H key/query heads   (paper 1.3B: 16)
    gdn_head_k_dim: int = 64  # d_k                 (paper: 128)
    gdn_head_v_dim: int = 64  # d_v                 (paper: 128)
    gdn_num_v_heads: int | None = None  # H_v for GQA value heads; None -> = num_heads
    gdn_chunk_size: int = 64  # chunkwise block size C (paper App.: 64).
    # Non-aligned training lengths use the recurrent core for their ragged tail.
    gdn_conv_size: int = 4  # short-conv kernel width
    gdn_expanded_erase: bool = False  # erase gate in [0,2] (neg-eigenvalue variant)

    # --- MLA full-attention layers (NoPE) — see multi_latent_attention/attention.py ---
    mla_num_q_heads: int = 8  # query heads
    mla_num_kv_heads: int = 2  # KV/latent heads (GQA); q_heads must be a multiple
    mla_head_dim: int = 64  # per-head latent (rank) width
    # Declared context cap: checked against the training seq_len and used as the
    # default size of the preallocated MLA latent cache in init_cache/generate.
    # (The MLA causal mask itself is built on the fly from the actual length.)
    max_seq_len: int = 512
    mla_gated: bool = True

    # --- Attention Residuals ---
    # "block" is the scalable paper variant, "full" attends to every prior sublayer
    # output, and "none" restores ordinary additive PreNorm residuals. Block size
    # counts token-mixer and channel-mixer sublayers (two per decoder layer).
    attnres_mode: str = "block"
    attnres_block_size: int = 4

    # --- Channel mixer (FFN) ---
    moe_d_ff: int = 512  # per-expert hidden width (paper: 1408 at 1.3B)
    moe_n_routed: int = 8  # number of routed experts E (paper: 256)
    moe_n_shared: int = 1  # always-on shared experts
    moe_top_k: int = 2  # experts activated per token (paper: 8)
    # Group-limited routing (DeepSeek-V3 / Kimi K2 "node-limited"): experts split
    # into moe_n_groups groups; each token draws its top-k only from its
    # moe_topk_groups best groups (at scale: bounds all-to-all traffic). 8 experts
    # in 4 groups, top-2 groups mirrors V3's half-the-groups ratio. Constraints:
    # moe_n_routed % moe_n_groups == 0 and moe_top_k <= moe_topk_groups * group size.
    # Set moe_n_groups = 1 to disable.
    moe_n_groups: int = 4
    moe_topk_groups: int = 2
    # LatentMoE routed width. The recommended paper compression ratio is 4x;
    # router inputs and shared experts remain at full d_model width.
    moe_latent_dim: int | None = 64

    rms_eps: float = 1e-5

    # --- Mixed precision ---
    # Matmul (compute) dtype for the projection Linears + MoE expert GEMMs. Master
    # weights are ALWAYS stored fp32 (param_dtype), and the numerically sensitive
    # parts stay fp32 regardless: the GDN-2 chunkwise core, RMSNorm, the router
    # softmax, and the loss. Set "bfloat16" on an H200; "float32" disables mixed
    # precision. Read from YAML as a string; use `.cdtype` for the resolved dtype.
    compute_dtype: str = "float32"

    def __post_init__(self):
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "full_attn_period": self.full_attn_period,
            "gdn_num_heads": self.gdn_num_heads,
            "gdn_head_k_dim": self.gdn_head_k_dim,
            "gdn_head_v_dim": self.gdn_head_v_dim,
            "gdn_chunk_size": self.gdn_chunk_size,
            "gdn_conv_size": self.gdn_conv_size,
            "mla_num_q_heads": self.mla_num_q_heads,
            "mla_num_kv_heads": self.mla_num_kv_heads,
            "mla_head_dim": self.mla_head_dim,
            "max_seq_len": self.max_seq_len,
            "moe_d_ff": self.moe_d_ff,
            "moe_n_routed": self.moe_n_routed,
            "moe_n_shared": self.moe_n_shared,
            "moe_top_k": self.moe_top_k,
            "moe_n_groups": self.moe_n_groups,
            "moe_topk_groups": self.moe_topk_groups,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.attnres_mode not in {"none", "full", "block"}:
            raise ValueError("attnres_mode must be 'none', 'full', or 'block'")
        if self.attnres_block_size < 1:
            raise ValueError("attnres_block_size must be positive")
        if self.moe_latent_dim is not None and not (
            1 <= self.moe_latent_dim <= self.d_model
        ):
            raise ValueError("moe_latent_dim must be in [1, d_model] or None")
        if self.gdn_num_v_heads is not None and self.gdn_num_v_heads < 1:
            raise ValueError("gdn_num_v_heads must be positive or None")
        gdn_v_heads = (
            self.gdn_num_heads
            if self.gdn_num_v_heads is None
            else self.gdn_num_v_heads
        )
        if gdn_v_heads < 1 or gdn_v_heads % self.gdn_num_heads != 0:
            raise ValueError(
                "gdn_num_v_heads must be positive and a multiple of gdn_num_heads"
            )
        if self.mla_num_q_heads % self.mla_num_kv_heads != 0:
            raise ValueError("mla_num_q_heads must be divisible by mla_num_kv_heads")
        if self.moe_n_routed % self.moe_n_groups != 0:
            raise ValueError("moe_n_routed must be divisible by moe_n_groups")
        if self.moe_topk_groups > self.moe_n_groups:
            raise ValueError("moe_topk_groups cannot exceed moe_n_groups")
        group_size = self.moe_n_routed // self.moe_n_groups
        if self.moe_top_k > self.moe_topk_groups * group_size:
            raise ValueError(
                "moe_top_k exceeds the experts available in selected groups"
            )
        if self.rms_eps <= 0:
            raise ValueError("rms_eps must be positive")
        if self.compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError("compute_dtype must be 'float32' or 'bfloat16'")

    @property
    def cdtype(self) -> jnp.dtype:
        return jnp.dtype(self.compute_dtype)


# --------------------------------------------------------------------------- #
#  One decoder block: pre-norm token mixer + pre-norm channel mixer, both residual.
#
#  The ONLY thing that varies across layers is the token mixer: GDN-2 (linear) on
#  most layers, MLA (full attention) on the 3:1 schedule. The channel mixer is a MoE
#  on every layer — this matches Kimi Linear, where the hybrid is in the *attention*,
#  not the FFN.
# --------------------------------------------------------------------------- #
class DecoderLayer(nnx.Module):
    def __init__(self, cfg: KimiLinearConfig, layer_idx: int, *, rngs: nnx.Rngs):
        # 3:1 schedule: this layer is full-attention iff it is the last of its period.
        self.is_full_attn = (layer_idx + 1) % cfg.full_attn_period == 0

        # Pre-norm before the token mixer (Fig. 2). RMSNorm reused from the GDN-2 layer.
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
        self.token_residual = (
            AttentionResidual(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
            if cfg.attnres_mode != "none"
            else None
        )

        if self.is_full_attn:
            # Full attention: NoPE gated MLA (absorbed/GQA form).
            mla_cls = (
                GatedMultiHeadLatentAttention
                if cfg.mla_gated
                else GroupedQueryLatentAttention
            )
            mla_kwargs = {} if cfg.mla_gated else {"gated": False}
            self.token_mixer = mla_cls(
                embed_dim=cfg.d_model,
                num_q_heads=cfg.mla_num_q_heads,
                num_kv_heads=cfg.mla_num_kv_heads,
                head_dim=cfg.mla_head_dim,
                compute_dtype=cfg.cdtype,
                rngs=rngs,
                **mla_kwargs,
            )
        else:
            # Linear attention: Gated DeltaNet-2 (the KDA substitute).
            self.token_mixer = GatedDeltaNet2(
                d_model=cfg.d_model,
                num_heads=cfg.gdn_num_heads,
                head_k_dim=cfg.gdn_head_k_dim,
                head_v_dim=cfg.gdn_head_v_dim,
                num_v_heads=cfg.gdn_num_v_heads,
                chunk_size=cfg.gdn_chunk_size,
                conv_size=cfg.gdn_conv_size,
                expanded_erase=cfg.gdn_expanded_erase,
                compute_dtype=cfg.cdtype,
                rngs=rngs,
            )

        # Pre-norm before the channel mixer.
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
        self.channel_residual = (
            AttentionResidual(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
            if cfg.attnres_mode != "none"
            else None
        )

        # Channel mixer: LatentMoE by default; setting moe_latent_dim=None restores
        # the original full-width routed experts for checkpoint compatibility.
        moe_cls = LatentMoE if cfg.moe_latent_dim is not None else GroupedGemmMoE
        moe_kwargs = (
            {"latent_dim": cfg.moe_latent_dim} if cfg.moe_latent_dim is not None else {}
        )
        self.channel_mixer = moe_cls(
            d_model=cfg.d_model,
            d_ff=cfg.moe_d_ff,
            n_routed=cfg.moe_n_routed,
            n_shared=cfg.moe_n_shared,
            top_k=cfg.moe_top_k,
            n_groups=cfg.moe_n_groups,
            topk_groups=cfg.moe_topk_groups,
            compute_dtype=cfg.cdtype,
            rngs=rngs,
            **moe_kwargs,
        )

    def token_delta(self, h: jax.Array) -> jax.Array:
        """Token-mixer sublayer output without applying a residual merge."""
        return self.token_mixer(self.norm1(h))

    def channel_delta(self, h: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """LatentMoE sublayer output without applying a residual merge."""
        return self.channel_mixer(self.norm2(h))

    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """x: [B, L, d_model] -> (x, aux_or_None).

        `aux` carries the MoE load-balancing diagnostics the training loop
        needs (aux loss + per-expert token counts for the router-bias update).
        """
        # --- token mixing (residual, pre-norm) ---
        x = x + self.token_delta(x)

        # --- channel mixing (residual, pre-norm) ---
        m, aux = self.channel_delta(x)
        x = x + m
        return x, aux

    def init_cache(self, batch_size: int, max_len: int, dtype=None):
        """Per-layer streaming cache: a GDN2Cache (linear layer) or MLACache (MLA)."""
        return self.token_mixer.init_cache(batch_size, max_len, dtype)

    def stream_token_delta(
        self, h: jax.Array, cache: GDN2Cache | MLACache
    ) -> tuple[jax.Array, GDN2Cache | MLACache]:
        """Streaming token-mixer delta without applying a residual merge."""
        h = self.norm1(h)
        if isinstance(cache, GDN2Cache) and isinstance(
            self.token_mixer, GatedDeltaNet2
        ):
            return self.token_mixer.step(h, cache)
        if isinstance(cache, MLACache) and isinstance(
            self.token_mixer, GroupedQueryLatentAttention
        ):
            return self.token_mixer.step(h, cache)
        raise ValueError(
            f"Cache type {type(cache)} does not match token mixer {type(self.token_mixer)}"
        )

    def step(
        self, x: jax.Array, cache: GDN2Cache | MLACache
    ) -> tuple[jax.Array, GDN2Cache | MLACache]:
        """Streaming forward for one block. x: [B, L, d_model] -> (x, new_cache).
        Only the token mixer is stateful; the channel mixer (MoE) is position-wise,
        so it needs no cache."""
        h, new_cache = self.stream_token_delta(x, cache)

        x = x + h
        m, _ = self.channel_delta(x)
        x = x + m
        return x, new_cache


# --------------------------------------------------------------------------- #
#  The full model.
# --------------------------------------------------------------------------- #
class KimiLinear(nnx.Module):
    """Decoder-only Kimi Linear LM with a GDN-2 linear-attention backbone."""

    def __init__(self, cfg: KimiLinearConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        # Token embedding table.
        self.embed = nnx.Embed(
            cfg.vocab_size, cfg.d_model, embedding_init=_XAVIER, rngs=rngs
        )

        # Stack of decoder blocks. NOTE: in Flax NNX a plain Python list of submodules
        # is not tracked as state — it must be wrapped in nnx.List(...).
        self.layers = nnx.List(
            [DecoderLayer(cfg, i, rngs=rngs) for i in range(cfg.n_layers)]
        )
        # The paper explicitly aggregates all completed depth sources before the
        # output head. This query is distinct from every pre-sublayer query.
        self.final_residual = (
            AttentionResidual(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
            if cfg.attnres_mode != "none"
            else None
        )

        # Final pre-head norm + untied LM head (Moonlight/DeepSeek do not tie weights;
        # to tie, drop lm_head and use `x @ self.embed.embedding[...].T` instead).
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
        self.lm_head = nnx.Linear(
            cfg.d_model,
            cfg.vocab_size,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=cfg.cdtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    @staticmethod
    def _validate_input_ids(input_ids: jax.Array, *, max_len: int | None = None):
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, length], got {input_ids.shape}"
            )
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids batch and sequence dimensions must be non-zero")
        if not jnp.issubdtype(input_ids.dtype, jnp.integer):
            raise TypeError(f"input_ids must use an integer dtype, got {input_ids.dtype}")
        if max_len is not None and input_ids.shape[1] > max_len:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds max_seq_len {max_len}"
            )

    def __call__(self, input_ids: jax.Array) -> tuple[jax.Array, dict[str, ArrayLike]]:
        """input_ids: int[B, L] -> (logits[B, L, vocab], aux).

        aux is ALWAYS returned (callers that don't need it just unpack `logits, _ =`):
            aux = {"aux_loss":   scalar, the MoE load-balancing loss summed over layers,
                   "group_sizes": int[n_layers, E], per-expert token counts per layer}.
        The training loop uses aux_loss (added to the CE loss) and group_sizes (to nudge
        each MoE layer's router bias); eval/inference paths simply ignore it.
        """
        self._validate_input_ids(input_ids, max_len=self.cfg.max_seq_len)
        aux_loss: ArrayLike = 0.0
        group_sizes: list[
            ArrayLike
        ] = []  # one [E] vector per MoE layer, in layer order

        x = self.embed(input_ids)  # [B, L, d_model]
        if self.cfg.attnres_mode == "none":
            for layer in self.layers:
                x, aux = layer(x)
                aux_loss = aux_loss + aux["aux_loss"]
                group_sizes.append(aux["group_sizes"])
        elif self.cfg.attnres_mode == "full":
            values = [x]
            for layer in self.layers:
                assert layer.token_residual is not None
                assert layer.channel_residual is not None

                h = layer.token_residual(values)
                values.append(layer.token_delta(h))

                h = layer.channel_residual(values)
                delta, aux = layer.channel_delta(h)
                values.append(delta)

                aux_loss = aux_loss + aux["aux_loss"]
                group_sizes.append(aux["group_sizes"])

            assert self.final_residual is not None
            x = self.final_residual(values)
        else:
            # Block AttnRes: completed blocks include the token embedding b_0.
            # Sublayer deltas are summed inside the current block; only completed
            # block summaries and the evolving partial sum are attended over.
            completed = [x]
            partial = None
            sublayer_idx = 0

            for layer in self.layers:
                assert layer.token_residual is not None
                assert layer.channel_residual is not None

                sources = completed if partial is None else [*completed, partial]
                h = layer.token_residual(sources)
                delta = layer.token_delta(h)
                partial = delta if partial is None else partial + delta
                sublayer_idx += 1
                if sublayer_idx % self.cfg.attnres_block_size == 0:
                    completed.append(partial)
                    partial = None

                sources = completed if partial is None else [*completed, partial]
                h = layer.channel_residual(sources)
                delta, aux = layer.channel_delta(h)
                partial = delta if partial is None else partial + delta
                sublayer_idx += 1
                if sublayer_idx % self.cfg.attnres_block_size == 0:
                    completed.append(partial)
                    partial = None

                aux_loss = aux_loss + aux["aux_loss"]
                group_sizes.append(aux["group_sizes"])

            if partial is not None:
                completed.append(partial)
            assert self.final_residual is not None
            x = self.final_residual(completed)

        x = self.norm_f(x)
        # Upcast logits to fp32 for a numerically stable softmax/cross-entropy under
        # bf16 compute (the lm_head matmul itself still runs in cfg.compute_dtype).
        logits = self.lm_head(x).astype(jnp.float32)  # [B, L, vocab]

        return logits, {"aux_loss": aux_loss, "group_sizes": jnp.stack(group_sizes)}

    # ----------------------------------------------------------------------- #
    #  Streaming / inference.  Each layer carries its own cache (GDN-2: fixed-size
    #  recurrent state + conv state; MLA: growing latent cache).  Reusing them makes
    #  generation O(1) per token for the linear layers instead of re-reading history.
    # ----------------------------------------------------------------------- #
    def init_cache(
        self, batch_size: int, max_len: int | None = None, dtype=None
    ) -> list:
        """Streaming caches for every layer. `max_len` (default cfg.max_seq_len) sizes
        the MLA latent buffers; GDN-2 layers ignore it (their state is fixed-size)."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        max_len = self.cfg.max_seq_len if max_len is None else max_len
        if max_len < 1:
            raise ValueError("max_len must be positive")
        dtype = self.cfg.cdtype if dtype is None else dtype
        return [layer.init_cache(batch_size, max_len, dtype) for layer in self.layers]

    def step(self, input_ids: jax.Array, caches: list) -> tuple[jax.Array, list]:
        """One streaming step. input_ids: int[B, L] (L = prompt length on prefill, or
        1 per decoded token). Returns (logits[B, L, vocab], new_caches)."""
        self._validate_input_ids(input_ids)
        new_caches = []

        if len(caches) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} layer caches, got {len(caches)}"
            )

        x = self.embed(input_ids)
        if self.cfg.attnres_mode == "none":
            for layer, cache in zip(self.layers, caches):
                x, new_cache = layer.step(x, cache)
                new_caches.append(new_cache)
        elif self.cfg.attnres_mode == "full":
            values = [x]
            for layer, cache in zip(self.layers, caches):
                assert layer.token_residual is not None
                assert layer.channel_residual is not None

                h = layer.token_residual(values)
                h, new_cache = layer.stream_token_delta(h, cache)
                new_caches.append(new_cache)
                values.append(h)

                h = layer.channel_residual(values)
                delta, _ = layer.channel_delta(h)
                values.append(delta)

            assert self.final_residual is not None
            x = self.final_residual(values)
        else:
            completed = [x]
            partial = None
            sublayer_idx = 0

            for layer, cache in zip(self.layers, caches):
                assert layer.token_residual is not None
                assert layer.channel_residual is not None

                sources = completed if partial is None else [*completed, partial]
                h = layer.token_residual(sources)
                delta, new_cache = layer.stream_token_delta(h, cache)
                new_caches.append(new_cache)
                partial = delta if partial is None else partial + delta
                sublayer_idx += 1
                if sublayer_idx % self.cfg.attnres_block_size == 0:
                    completed.append(partial)
                    partial = None

                sources = completed if partial is None else [*completed, partial]
                h = layer.channel_residual(sources)
                delta, _ = layer.channel_delta(h)
                partial = delta if partial is None else partial + delta
                sublayer_idx += 1
                if sublayer_idx % self.cfg.attnres_block_size == 0:
                    completed.append(partial)
                    partial = None

            if partial is not None:
                completed.append(partial)
            assert self.final_residual is not None
            x = self.final_residual(completed)

        x = self.norm_f(x)
        return self.lm_head(x).astype(jnp.float32), new_caches

    def generate(
        self, prompt_ids: jax.Array, max_new_tokens: int, max_len: int | None = None
    ) -> jax.Array:
        """Greedy autoregressive decode that REUSES each layer's state across steps.
        prompt_ids: int[B, P]. Returns the continuation int[B, max_new_tokens].

        Prefill consumes the whole prompt in one step (filling every layer's cache) —
        the GDN-2 layers push all whole chunks of the prompt through their PARALLEL
        chunkwise core and only the ragged tail through the recurrence, so prefill
        cost scales with P/chunk_size sequential steps, not P. Each decode step then
        feeds back ONE token and carries the caches forward — the GDN-2 layers via
        their fixed-size recurrent state, the MLA layers via the growing latent
        cache. The decode loop runs through `_decode_step`, a module-level nnx.jit
        function: it compiles once per (batch size, cache length) and every further
        token — across generate() calls too — reuses the trace."""
        self._validate_input_ids(prompt_ids)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")

        B, P = prompt_ids.shape
        if max_new_tokens == 0:
            return jnp.empty((B, 0), dtype=prompt_ids.dtype)

        # Default the cache length to the config's declared context cap when the
        # request fits inside it: a FIXED cache shape lets _decode_step reuse its
        # compiled trace across generate() calls with different prompt lengths
        # (e.g. a chat loop) instead of recompiling for every P + max_new_tokens.
        required_cache_len = P + max_new_tokens - 1
        if max_len is None:
            max_len = max(self.cfg.max_seq_len, required_cache_len)
        elif max_len < required_cache_len:
            raise ValueError(
                f"max_len={max_len} is too small; generation requires at least "
                f"{required_cache_len} cache positions"
            )

        caches = self.init_cache(B, max_len)
        logits, caches = self.step(prompt_ids, caches)  # prefill the prompt
        next_tok = jnp.argmax(logits[:, -1:], axis=-1)  # [B, 1] greedy
        outs = [next_tok]

        for _ in range(max_new_tokens - 1):
            next_tok, caches = _decode_step(self, next_tok, caches)
            outs.append(next_tok)

        return jnp.concatenate(outs, axis=1)  # [B, max_new_tokens]


# --------------------------------------------------------------------------- #
#  Jitted greedy decode step, shared by every generate() call.
#
#  During decoding everything is shape-constant — the weights, the fixed-size
#  GDN-2 states, the preallocated MLA latent buffers (position is a TRACED int32,
#  so advancing it never retraces), and L=1 — so this compiles ONCE per (batch
#  size, cache length) and each further token replays the compiled trace.
#  Module-level on purpose: nnx.jit keys its compilation cache on the function
#  object, so a wrapper created inside generate() would recompile every call.
# --------------------------------------------------------------------------- #
@nnx.jit
def _decode_step(
    model: KimiLinear, tok: jax.Array, caches: list
) -> tuple[jax.Array, list]:
    """One greedy decode step: tok int[B, 1] -> (next greedy token int[B, 1], caches)."""
    logits, caches = model.step(tok, caches)
    return jnp.argmax(logits[:, -1:], axis=-1), caches


def count_params(model: nnx.Module) -> int:
    """Total number of trainable parameters (sum of nnx.Param leaf sizes)."""
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))))
