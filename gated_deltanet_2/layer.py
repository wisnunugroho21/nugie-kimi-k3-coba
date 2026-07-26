"""
Gated DeltaNet-2 token-mixer layer in Flax NNX, ANNOTATED against the paper
(arXiv:2605.22791): Section 3.5 (block design) and Appendix C.1 (layer
parameterization), with supporting equations 11, 12, 85, 86 and the numerical
notes in Appendix D.

Block design (Fig. 1 right; Sec. 3.5 "Gated DeltaNet-2 token mixer"):
  q,k = L2norm(SiLU(ShortConv(Linear(x))))      # key-side paths + L2 norm (Sec. 3.5, App. D.2)
  v   =        SiLU(ShortConv(Linear(x)))        # value path (Sec. 3.5; Fig. 1 caption)
  g   = -exp(a) ⊙ softplus(Proj_f(x) + delta)    # log-decay, fp32 (Eq. 12 / 86, App. D.1)
  b   = sigmoid(Proj_b(x))                       # erase gate (Eq. 11 / 85); x2 if neg-eigenvalue
  w   = sigmoid(Proj_w(x))                       # write gate (Eq. 11 / 85)
  O   = chunkwise_gated_delta_rule_2(q,k,v,g,b,w, state)   # Gated Delta Rule-2 (Eq. 10)
  out = Linear_o( RMSNorm(O) * SiLU(Proj_g(x)) )  # gated RMSNorm + out proj (Sec. 3.5, App. D.5)

Grouped value heads (Sec. 3.5 last sentence / App. C.1): with num_v_heads = G*num_heads,
the key-side tensors q, k, the log-decay g, and b are shared across the G value heads
of each group; v and w live on the value-head axis. App. C.1 phrases the sharing as
"repeated across the value-head group"; this implementation realizes the SAME math
without the G× duplication: the recurrence is linear along the value axis, so each
group folds into one recurrence of value width G·d_v, and the key-side cumsums,
score matrices, and triangular solve are computed once per key head instead of once
per value head. The public state layout stays per value head: [B, Hv, d_k, d_v].

Scope: this is the recurrent TOKEN MIXER only (Fig. 1 right). The recurrent model
(Sec. 3.5 "Model families") stacks [this + MLP]; the hybrid model inserts
Sliding-Window Attention after it, repeating the cell [GDN-2, MLP, SWA, MLP]
(Fig. 1 left). Those wrappers are not implemented here.

Parameterization notes (App. C.1 / D.5 and the paper-linked reference code):
  * 'a' is stored per key HEAD ([H]) and broadcast across the d_k channels of
    that head; the bias δ is stored per key channel ([H·d_k]) — both exactly
    as App. C.1 specifies.
  * All Linear kernels use Xavier-uniform init with gain 2^{-2.5}; biases are
    zero when present (App. D.5).
  * a and δ follow the paper-linked reference initialization:
    a = log U(1, 16), and δ = softplus⁻¹(dt) with dt log-uniform in
    [1e-3, 1e-1]. The PDF specifies their shapes and use, but not this init.
  * The projection factorization and convolution details that the PDF leaves
    implicit follow the reference code: Proj_f and the output gate use a
    head_v_dim bottleneck; b, w, f, and the depthwise convolutions are biasless.
    The short-convolution width is 4 by default.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from gated_deltanet_2.core import (
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)

F32 = jnp.float32

# App. D.5: Xavier-uniform init with gain 2^{-2.5} (variance_scaling scale = gain² =
# 2^{-5}), replacing Flax NNX's default Linear kernel init. Biases stay at zero (the
# NNX default); the decay parameters a and δ have their own init (see __init__).
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


# --------------------------------------------------------------------------- #
#  Inference cache for streaming (incremental) decode.
#
#  Linear attention's headline property: the entire history collapses into a
#  FIXED-SIZE recurrent state S [B,Hv,dk,dv] — it does NOT grow with sequence
#  length (contrast a softmax KV-cache). To decode token-by-token we just carry S
#  across calls.  The short causal conv ALSO has a (kernel_size)-wide receptive
#  field, so we must additionally cache its last (kernel_size-1) inputs — otherwise
#  the first streamed tokens would see wrong, zero-padded context.  That is the
#  WHOLE state of a GDN-2 layer; both pieces are fixed-size.
# --------------------------------------------------------------------------- #
class GDN2Cache(NamedTuple):
    recurrent_state: jax.Array  # [B, Hv, dk, dv]  the gated-delta-rule memory S
    q_conv: jax.Array  # [B, conv_size-1, H*dk]   last inputs to the q short-conv
    k_conv: jax.Array  # [B, conv_size-1, H*dk]   last inputs to the k short-conv
    v_conv: jax.Array  # [B, conv_size-1, Hv*dv]  last inputs to the v short-conv


class RMSNorm(nnx.Module):
    """Plain RMSNorm used by the recurrent output stage.

    Takes no rngs: its only parameter is the deterministic all-ones gain.
    """

    def __init__(self, dim: int, *, eps: float = 1e-5):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        xf = x.astype(F32)
        mean = jnp.mean(xf * xf, axis=-1, keepdims=True)
        rms = jax.lax.rsqrt(mean + self.eps)

        # Scale by the fp32 weight BEFORE the downcast — casting first would
        # promote the result right back to fp32 and waste the cast.
        return (xf * rms * self.weight[...]).astype(x.dtype)


class LowRankLinear(nnx.Module):
    """Two-layer projection used by the reference decay and output-gate paths.

    Computes ``up(down(x))`` with a narrow per-value-head bottleneck. Both
    kernels use the paper's Xavier-uniform initialization; only the up
    projection optionally has a bias.
    """

    def __init__(
        self,
        in_features: int,
        rank: int,
        out_features: int,
        *,
        use_bias: bool,
        compute_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.down = nnx.Linear(
            in_features,
            rank,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.up = nnx.Linear(
            rank,
            out_features,
            use_bias=use_bias,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.up(self.down(x))


class GatedRMSNorm(nnx.Module):
    """Head-wise RMSNorm of the recurrent output, gated by a SiLU gate.

    Implements the output stage in Sec. 3.5 and Appendix D.5:

        SiLU(W↑g W↓g x) ⊙ RMSNorm(O)

    The gate is produced INSIDE the norm from the block input x, so the call site in
    your layer collapses to `O = self.o_norm(O_heads, x)` (no separate gate_proj).
    """

    def __init__(
        self,
        head_dim: int,
        d_model: int,
        inner_dim: int,
        gate_rank: int,
        *,
        eps: float = 1e-5,
        compute_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.norm = RMSNorm(head_dim, eps=eps)
        self.gate = LowRankLinear(
            d_model,
            gate_rank,
            inner_dim,
            use_bias=True,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )

    def __call__(self, O_heads: jax.Array, x: jax.Array) -> jax.Array:
        """O_heads: [B, L, Hv, dv]   x: [B, L, d_model]  ->  [B, L, Hv*dv]."""
        B, L, Hv, dv = O_heads.shape

        o = O_heads.astype(F32)  # [B,L,Hv,dv] -> fp32 for RMSNorm
        o = self.norm(o)  # head-wise RMSNorm

        g = jax.nn.silu(self.gate(x).astype(F32))
        g = g.reshape(B, L, Hv, dv)

        return (o * g).reshape(B, L, Hv * dv)


class ShortConv(nnx.Module):
    """Causal depthwise 1-D convolution — the 'Conv' boxes in Fig. 1 (Sec. 3.5).

    The paper says only "short causal convolution"; the kernel width (default 4)
    is an implementation choice, as in the Mamba/GatedDeltaNet lineage.

    nnx.Conv is channels-last ([B, L, C]) and owns the kernel, so the manual
    NCW transposes and the raw conv call disappear. Padding is fixed at construction,
    so we run the conv in 'VALID' mode and keep the causal left-context / streaming
    state ourselves (state = the trailing kernel_size-1 inputs).
    """

    def __init__(self, channels: int, kernel_size: int = 4, *, rngs: nnx.Rngs):
        self.channels = channels
        self.kernel_size = kernel_size
        # The PDF is silent on convolution initialization. Keep NNX's default
        # kernel initialization and match the paper-linked reference's
        # biasless depthwise convolution.
        self.conv = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=(kernel_size,),
            feature_group_count=channels,  # depthwise: one filter per channel
            padding="VALID",  # left context supplied manually below
            use_bias=False,
            rngs=rngs,
        )

    def _apply(
        self, x: jax.Array, conv_state: jax.Array | None
    ) -> tuple[jax.Array, jax.Array]:
        """Shared conv core. `conv_state` is the previous (kernel_size-1) inputs used
        as left context, or None on the full/training path (pad with zeros == the
        causal left-pad). Returns (y: [B, L, C], new_state: [B, kernel_size-1, C])."""
        B, _, C = x.shape
        kc = self.kernel_size - 1

        left = jnp.zeros((B, kc, C), x.dtype) if conv_state is None else conv_state
        xc = jnp.concatenate([left, x], axis=1)  # [B, kc+L, C]
        new_state = xc[:, xc.shape[1] - kc :, :]  # last kc inputs -> next context

        y = self.conv(xc)  # VALID: (kc+L)-(kc+1)+1 = L
        return y, new_state  # [B, L, C]

    def __call__(
        self, x: jax.Array
    ) -> jax.Array:  # full-sequence (training) path; left context = zeros
        y, _ = self._apply(x, conv_state=None)
        return y

    def step(
        self, x: jax.Array, conv_state: jax.Array
    ) -> tuple[jax.Array, jax.Array]:  # streaming path; carry the left context in/out
        # Single-token decode fast path: the conv output at ONE position is
        # just a dot of each channel's kernel with its (kernel_size)-token
        # window — cheaper than dispatching a general convolution. Same math
        # as _apply (verified by the decode test in test_layer.py).
        if x.shape[1] == 1:
            window = jnp.concatenate([conv_state, x], axis=1)  # [B, W, C]
            kernel = self.conv.kernel[...][:, 0, :]  # depthwise [W, 1, C] -> [W, C]
            y = jnp.einsum("bwc,wc->bc", window, kernel)[:, None, :]
            if self.conv.bias is not None:
                y = y + self.conv.bias[...]
            return y, window[:, 1:, :]
        return self._apply(x, conv_state)


class GatedDeltaNet2(nnx.Module):
    """Gated DeltaNet-2 recurrent token mixer (Fig. 1 right; Sec. 3.5 / App. C.1)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,  # H key heads; App. E.1 uses H=16 at 1.3B
        head_k_dim: int = 128,  # d_k; App. E.1 uses 128
        head_v_dim: int = 128,  # d_v; App. E.1 uses 128
        num_v_heads: int | None = None,  # H_v for GQA; defaults to H (App. C.1)
        chunk_size: int = 64,  # C; App. C.2 fixes C = 64
        conv_size: int = 4,
        expanded_erase: bool = False,  # erase gate in [0,2] (neg-eigenvalue variant; Sec. 3.1, App. C.1)
        compute_dtype: jnp.dtype = jnp.float32,
        core: str = "centered",  # core.py chunkwise core; "subchunking"/"pairwise"
        #   have no decay-range limit if the learned decay outgrows the centered
        #   core's per-chunk |G_C| ~ 176 (see chunkwise_gated_delta_rule_2)
        sub_chunk_size: int = 16,  # c for core="subchunking"; ignored otherwise
        *,
        rngs: nnx.Rngs,
    ):
        if d_model <= 0 or num_heads <= 0 or head_k_dim <= 0 or head_v_dim <= 0:
            raise ValueError(
                "d_model, num_heads, head_k_dim, and head_v_dim must be positive"
            )
        if num_v_heads is not None and num_v_heads <= 0:
            raise ValueError("num_v_heads must be positive when provided")
        if chunk_size <= 0 or conv_size <= 0:
            raise ValueError("chunk_size and conv_size must be positive")
        if sub_chunk_size <= 0:
            raise ValueError("sub_chunk_size must be positive")

        # Projection matmuls use compute_dtype (bf16 on H200). The recurrent core
        # upcasts to fp32, and the f_proj result is cast to fp32 before the
        # log-decay activation, matching Appendix D.1/D.3 and the reference code.
        self.compute_dtype = compute_dtype
        self.d_model = d_model
        self.H = num_heads
        self.Hv = num_heads if num_v_heads is None else num_v_heads

        if self.Hv % self.H:
            raise ValueError("num_v_heads must be a multiple of num_heads")

        self.group = self.Hv // self.H  # G, value-head group size (App. C.1)
        self.dk = head_k_dim
        self.dv = head_v_dim
        self.chunk_size = chunk_size
        self.conv_size = conv_size  # kernel width; sizes the streaming conv cache
        self.expanded_erase = expanded_erase
        self.core = core
        self.sub_chunk_size = sub_chunk_size

        # App. C.1 projection shapes: erase/key side -> H·d_k, write/value side -> H_v·d_v.
        k_proj_dim = self.H * self.dk  # q, k, b live on the key-head axis
        v_proj_dim = self.Hv * self.dv  # v, w live on the value-head axis

        # Linear projections feeding the SiLU/conv paths (Sec. 3.5; Fig. 1 'Linear' boxes).
        self.q_proj = nnx.Linear(
            d_model,
            k_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            d_model,
            k_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            d_model,
            v_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.b_proj = nnx.Linear(
            d_model,
            k_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )  # Proj_b, Eq. 85: b = σ(Proj_b x)
        self.w_proj = nnx.Linear(
            d_model,
            v_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )  # Proj_w, Eq. 85: w = σ(Proj_w x)
        self.f_proj = LowRankLinear(
            d_model,
            self.dv,
            k_proj_dim,
            use_bias=False,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )  # Proj_f, Eq. 86: d_model -> d_v -> H·d_k

        # Short causal convs on q, k, v (App. C.1: "short-convolutional projections for q, k, v").
        self.q_conv = ShortConv(k_proj_dim, conv_size, rngs=rngs)
        self.k_conv = ShortConv(k_proj_dim, conv_size, rngs=rngs)
        self.v_conv = ShortConv(v_proj_dim, conv_size, rngs=rngs)

        # Log-decay parameters (Eq. 12 / 86; App. C.1): 'a' is stored PER KEY
        # HEAD ([H]) and broadcast across the d_k channels of that head; the
        # bias δ is stored per key channel ([H·d_k]) and added pre-softplus.
        # Init follows the paper-linked reference implementation:
        #   a = log U(1, 16)  -> exp(a) ∈ [1, 16], a spread of per-head
        #                        base forgetting rates;
        #   δ = softplus⁻¹(dt), dt log-uniform in [1e-3, 1e-1] -> the decay
        #       magnitude at init (where Proj_f x ≈ 0) is small: long memory,
        #       with a spread of time scales across channels.
        self.A_log = nnx.Param(
            jnp.log(jax.random.uniform(rngs.params(), (self.H,), F32, 1.0, 16.0))
        )  # 'a' in -exp(a)·softplus(·)
        dt = jnp.exp(
            jax.random.uniform(
                rngs.params(),
                (self.H * self.dk,),
                F32,
                jnp.log(1e-3),
                jnp.log(1e-1),
            )
        )
        self.dt_bias = nnx.Param(dt + jnp.log(-jnp.expm1(-dt)))  # δ = softplus⁻¹(dt)

        # Output gate + gated RMSNorm + output projection (Sec. 3.5 / App. D.5).
        self.o_norm = GatedRMSNorm(
            head_dim=self.dv,
            d_model=d_model,
            inner_dim=self.Hv * self.dv,
            gate_rank=self.dv,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )

        self.o_proj = nnx.Linear(
            v_proj_dim,
            d_model,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )  # back to d_model

        # App. D.5: every Linear kernel above uses Xavier-uniform init, gain 2^{-2.5}
        # (_XAVIER); biases are zero when present. The decay parameters a and δ use
        # the Gated DeltaNet family init described above.

    def _split_k(self, x: jax.Array, B: int, L: int) -> jax.Array:
        # Head reshaping for key-side tensors (App. C.1: "followed by head reshaping").
        return x.reshape(B, L, self.H, self.dk).swapaxes(1, 2)  # [B,H,L,dk]

    def _split_v(self, x: jax.Array, B: int, L: int) -> jax.Array:
        # Head reshaping for value-side tensors, GROUPED for GQA: the G value
        # heads owned by one key head are contiguous in the flat projection,
        # so folding them into one value axis of width G·d_v is a plain
        # reshape. For G = 1 this is the ordinary [B, Hv, L, dv] head split.
        return x.reshape(B, L, self.H, self.group * self.dv).swapaxes(1, 2)

    # -- recurrent-state layout converters (GQA) ---------------------------- #
    # Public layout: one [dk, dv] memory per VALUE head -> [B, Hv, dk, dv]
    # (App. C.1 semantics; what GDN2Cache and initial_state use).
    # Internal layout: the G group members concatenated along the value axis
    # -> [B, H, dk, G·dv], what the folded recurrence consumes. Bijective
    # reshapes; for G = 1 both layouts coincide.
    def _state_in(self, S: jax.Array) -> jax.Array:
        B = S.shape[0]
        return (
            S.reshape(B, self.H, self.group, self.dk, self.dv)
            .swapaxes(2, 3)  # [B, H, dk, G, dv]
            .reshape(B, self.H, self.dk, self.group * self.dv)
        )

    def _state_out(self, S: jax.Array) -> jax.Array:
        B = S.shape[0]
        return (
            S.reshape(B, self.H, self.dk, self.group, self.dv)
            .swapaxes(2, 3)  # [B, H, G, dk, dv]
            .reshape(B, self.Hv, self.dk, self.dv)
        )

    def _project(
        self, x: jax.Array, conv_states: tuple[jax.Array, jax.Array, jax.Array] | None
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        tuple[jax.Array, jax.Array, jax.Array] | None,
    ]:
        """Shared front-end used by BOTH the training and streaming paths:
        Linear -> ShortConv -> SiLU -> head split -> L2 norm, plus the log-decay g
        and the channel-wise gates b, w.  `conv_states` is None on the full/training
        path, or a (q, k, v) tuple of conv caches when streaming.  Returns
        (q, k, v, g, b, w) on the KEY-head axis — v and w grouped to value
        width G·d_v (see the GQA note below) — plus the updated conv states
        (or None)."""
        B, L, _ = x.shape

        # q,k,v paths: Linear -> ShortConv -> SiLU (Sec. 3.5; Fig. 1 caption).
        if conv_states is None:  # full/training: conv pads with zeros (causal)
            q = self.q_conv(self.q_proj(x))
            k = self.k_conv(self.k_proj(x))
            v = self.v_conv(self.v_proj(x))
            new_conv = None
        else:  # streaming: conv uses the cached left context and returns a new one
            qcs, kcs, vcs = conv_states
            q, qcs = self.q_conv.step(self.q_proj(x), qcs)
            k, kcs = self.k_conv.step(self.k_proj(x), kcs)
            v, vcs = self.v_conv.step(self.v_proj(x), vcs)
            new_conv = (qcs, kcs, vcs)

        q, k, v = jax.nn.silu(q), jax.nn.silu(k), jax.nn.silu(v)

        q = self._split_k(q, B, L)
        k = self._split_k(k, B, L)
        v = self._split_v(v, B, L)

        # L2-normalize q, k per head (Sec. 3.5 "L2 normalization applied to q_t
        # and k_t"; App. D.2), as x·rsqrt(‖x‖² + ε) — one fused rsqrt instead
        # of a sqrt and a divide; ε guards the all-zero rows SiLU can produce.
        q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)

        # Log-decay branch, computed in fp32 outside the kernel (Eq. 12 / 86; App. C.1 / D.1).
        #   g_t = -exp(a) ⊙ softplus(Proj_f(x_t) + δ),  then α_t = exp(g_t) inside the core.
        f_p = self.f_proj(x).astype(jnp.float32)  # [B,L,H*dk]  Proj_f(x) in Eq. 86
        d_t = self.dt_bias[...].astype(
            jnp.float32
        )  # [H*dk]  per-channel bias δ, Eq. 86
        a_l = self.A_log[...].astype(jnp.float32)  # [H]  'a', per key head (App. C.1)

        f = self._split_k(f_p + d_t, B, L)  # Proj_f(x)+δ -> [B,H,L,dk]
        a = jnp.exp(a_l)[None, :, None, None]  # exp(a), broadcast over the d_k channels
        g = -a * jax.nn.softplus(f)  # [B,H,L,dk] ≤ 0  (Eq. 86)

        # Channel-wise gates (Eq. 11 / 85).
        b = jax.nn.sigmoid(self.b_proj(x))  # b = σ(Proj_b x) ∈ [0,1]^{d_k}
        b = self._split_k(b, B, L)

        if self.expanded_erase:
            b = 2.0 * b  # neg-eigenvalue variant: scale ONLY b to [0,2] (Sec. 3.1)

        w = jax.nn.sigmoid(self.w_proj(x))  # w = σ(Proj_w x) ∈ [0,1]^{d_v}
        w = self._split_v(w, B, L)

        # GQA (Sec. 3.5 / App. C.1): q, k, g, b are shared by the G value heads
        # of each group. App. C.1 phrases this as repeating the key-side tensors
        # to Hv heads; here NO repeat happens — the recurrence is LINEAR along
        # the value axis, so the G group members (which share every key-side
        # factor) are folded into one recurrence of value width G·d_v: v and w
        # arrived grouped from _split_v, and q, k, g, b stay on the H key-head
        # axis. Algebraically identical to the repeat formulation (verified in
        # test_layer.py), but the key-side cumsums, T/A_qk, and the triangular
        # solve are computed once per key head instead of G times.
        return q, k, v, g, b, w, new_conv

    def _output(self, o: jax.Array, x: jax.Array) -> jax.Array:
        """Gated RMSNorm + output projection (Sec. 3.5 / App. D.5).

        o arrives GROUPED from the core, [B, H, L, G·dv]; ungrouping to
        [B, L, Hv, dv] restores the per-VALUE-head axis so the RMSNorm
        normalizes each value head's dv channels separately."""
        B, _, L, _ = o.shape
        o = o.swapaxes(1, 2).reshape(B, L, self.Hv, self.dv)  # ungroup value heads
        o = self.o_norm(o, x).astype(
            x.dtype
        )  # SiLU output gate computed inside, from x (Sec. 3.5 / App. D.5)

        return self.o_proj(o)  # project back to d_model

    def _run_recurrence(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        b: jax.Array,
        w: jax.Array,
        S: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Run a chunk-aligned prefix followed by an exact recurrent tail.

        This gives training, prefill, and decode one recurrence dispatcher and
        removes the public layer's former sequence-length divisibility
        restriction. Inputs use the grouped internal head layout.
        """
        L = q.shape[2]
        if L < 1:
            raise ValueError("GatedDeltaNet2 requires a non-empty sequence")

        n_full = (L // self.chunk_size) * self.chunk_size
        outs = []

        if n_full:
            o_head, S = chunkwise_gated_delta_rule_2(
                q[:, :, :n_full],
                k[:, :, :n_full],
                v[:, :, :n_full],
                g[:, :, :n_full],
                b[:, :, :n_full],
                w[:, :, :n_full],
                S,
                chunk_size=self.chunk_size,
                core=self.core,
                sub_chunk_size=self.sub_chunk_size,
            )
            outs.append(o_head)

        if n_full < L:
            o_tail, S = recurrent_gated_delta_rule_2(
                q[:, :, n_full:],
                k[:, :, n_full:],
                v[:, :, n_full:],
                g[:, :, n_full:],
                b[:, :, n_full:],
                w[:, :, n_full:],
                S,
            )
            outs.append(o_tail)

        o = outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=2)
        return o, S

    def __call__(
        self,
        x: jax.Array,
        initial_state: jax.Array | None = None,
        return_state: bool = False,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Full-sequence forward using a chunkwise prefix and recurrent tail.

        x: [B, L, d_model] -> out: [B, L, d_model], or (out, S_final) with
        return_state=True. Any L >= 1 is supported.

        `initial_state` / the returned S_final use the public per-value-head
        layout [B, Hv, dk, dv], so state can be carried across segment calls
        (truncated-BPTT-style training: pass S_final of one segment — usually
        via jax.lax.stop_gradient — as initial_state of the next). CAVEAT:
        this carries the RECURRENT memory only; the short-conv left context is
        not part of it, so the first conv_size-1 tokens of a continued segment
        see zero-padding instead of the true previous tokens. For exact
        continuation (inference) use `step`, whose cache carries both.

        `attention_mask`, when provided, has shape [B, L], with nonzero entries
        marking valid tokens. Masked inputs are zeroed before the causal
        convolution, and their recurrence is made an exact no-op. This keeps
        both their output and their contribution to S_final zero, so right
        padding is safe when carrying state across calls.
        """
        B, L, _ = x.shape
        if L < 1:
            raise ValueError("GatedDeltaNet2 requires a non-empty sequence")

        if attention_mask is not None:
            if attention_mask.shape != (B, L):
                raise ValueError(
                    f"attention_mask must have shape {(B, L)}, got {attention_mask.shape}"
                )
            valid = attention_mask.astype(jnp.bool_)
            x_project = jnp.where(valid[..., None], x, jnp.zeros((), x.dtype))
        else:
            valid = None
            x_project = x

        q, k, v, g, b, w, _ = self._project(x_project, conv_states=None)

        if valid is not None:
            # A masked token must neither decay nor edit memory. g=0 makes
            # alpha=1, while k=0 removes both erase and write terms (Eq. 9).
            m = valid[:, None, :, None]
            q = jnp.where(m, q, 0)
            k = jnp.where(m, k, 0)
            v = jnp.where(m, v, 0)
            g = jnp.where(m, g, 0)
            b = jnp.where(m, b, 0)
            w = jnp.where(m, w, 0)

        if initial_state is None:
            S0 = jnp.zeros((B, self.H, self.dk, self.group * self.dv), jnp.float32)
        else:
            S0 = self._state_in(initial_state)

        o, S_final = self._run_recurrence(q, k, v, g, b, w, S0)

        out = self._output(o, x)
        if valid is not None:
            out = jnp.where(valid[..., None], out, 0)
        if return_state:
            return out, self._state_out(S_final)
        return out

    # ----------------------------------------------------------------------- #
    #  Streaming / inference.  Same math, threading the fixed-size state in -> out.
    #  `step` picks the fastest core for the given length: the chunk-aligned prefix
    #  of the input goes through the PARALLEL chunkwise core (fast prefill for long
    #  prompts), the ragged tail — which includes the L=1 decode step — through the
    #  RECURRENT core (no chunk-size divisibility constraint). One method serves
    #  both phases of decoding:
    #     prefill: out, cache = layer.step(prompt, layer.init_cache(B, ...))
    #     decode : out, cache = layer.step(one_token, cache)   # repeat
    # ----------------------------------------------------------------------- #
    def init_cache(
        self, batch_size: int, max_len: int | None = None, dtype=None
    ) -> GDN2Cache:
        """Empty streaming cache. `max_len` is accepted for interface parity with
        attention caches but UNUSED here — the GDN-2 state is fixed-size,
        independent of sequence length (the point of linear attention).

        The conv caches default to the layer's compute_dtype: they hold pre-conv
        projection outputs, which are produced in that dtype, so a mismatched
        cache (e.g. fp32 against bf16 activations) would silently promote the
        whole conv path at every decode step. The recurrent state S stays fp32
        regardless (App. D.3)."""
        kc = self.conv_size - 1
        dtype = dtype or self.compute_dtype
        return GDN2Cache(
            recurrent_state=jnp.zeros(
                (batch_size, self.Hv, self.dk, self.dv), jnp.float32
            ),
            q_conv=jnp.zeros((batch_size, kc, self.H * self.dk), dtype),
            k_conv=jnp.zeros((batch_size, kc, self.H * self.dk), dtype),
            v_conv=jnp.zeros((batch_size, kc, self.Hv * self.dv), dtype),
        )

    def step(self, x: jax.Array, cache: GDN2Cache) -> tuple[jax.Array, GDN2Cache]:
        """Streaming forward. x: [B, L, d_model] (L>=1). Returns (out, new_cache).

        The length is split as L = n_full + tail with n_full = (L // C)·C: the
        chunk-aligned prefix runs through the parallel CHUNKWISE core, the tail
        through the token-by-token RECURRENT core. Both compute the exact same
        recurrence and thread the same fixed-size state, so the split point is
        invisible in the output (verified in test_layer.py). Decode steps
        (L=1 < C) take the recurrent path only, as before.

        The prefill win is in SEQUENTIAL DEPTH: L/C scan steps instead of L,
        which is what dominates on accelerators. On CPU the recurrent scan is
        already compute-bound and the chunkwise core's pairwise-ratio tensor
        costs O(L·C·dk), so there the chunkwise path only wins for small C."""
        if x.shape[1] < 1:
            raise ValueError("GatedDeltaNet2.step requires a non-empty sequence")

        q, k, v, g, b, w, new_conv = self._project(
            x, conv_states=(cache.q_conv, cache.k_conv, cache.v_conv)
        )
        # We passed real conv_states, so _project always returns updated ones here
        # (it only returns None on the full-sequence/training path) — assert narrows
        # the tuple|None type for the checker and documents the invariant.
        assert new_conv is not None
        qcs, kcs, vcs = new_conv

        # Cache holds the public per-value-head layout; the cores consume the
        # grouped one (see _state_in/_state_out).
        S = self._state_in(cache.recurrent_state)
        o, S = self._run_recurrence(q, k, v, g, b, w, S)
        return self._output(o, x), GDN2Cache(self._state_out(S), qcs, kcs, vcs)
