# Mini Kimi K3 in JAX + Flax NNX

This repository is a small, readable implementation of the **text backbone** in
the paper *Kimi K3: Open Frontier Intelligence*. It preserves the model's main
ideas while shrinking the dimensions enough to run on a laptop.

The code follows the paper and Moonshot AI's
[official Kimi K3 release](https://huggingface.co/moonshotai/Kimi-K3):

- **Hybrid attention:** three Kimi Delta Attention (KDA) layers followed by one
  global Gated MLA layer, plus a final MLA layer.
- **KDA:** causal ShortConv, L2-normalized queries and keys, the delta-rule
  recurrence, lower-bounded channel-wise decay, and a full-rank output gate.
- **Gated MLA:** low-rank query and KV representations, NoPE causal attention,
  and a full-rank output gate.
- **Block Attention Residuals:** learned attention over completed block
  summaries and the current block's partial sum.
- **Stable LatentMoE:** full-width shared experts, narrow routed experts,
  sigmoid Top-k routing, routed-branch RMSNorm, SiTU-GLU, and exact local
  Quantile Balancing.
- **Published shape:** `KimiK3Config.paper()` records the official 93-layer,
  896-expert configuration. Do not instantiate it on a workstation.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m examples.tiny_forward
pytest
```

The basic API is intentionally short:

```python
import jax.numpy as jnp
from flax import nnx

from kimi_k3 import KimiK3, KimiK3Config

config = KimiK3Config.tiny()
model = KimiK3(config, rngs=nnx.Rngs(0))
token_ids = jnp.array([[1, 2, 3, 4]])
logits = model(token_ids)  # [batch, sequence, vocabulary]
```

Every important class and equation has a docstring or nearby explanatory
comment. Start with [`KimiK3`](kimi_k3/model.py), then read
`DecoderLayer`, `KimiDeltaAttention`, `GatedMLA`, and `StableLatentMoE`.

## What is intentionally simplified?

This is an **educational reimplementation**, not a checkpoint-compatible or
production inference engine.

| Paper / official model | This repository |
| --- | --- |
| Chunkwise fused KDA kernels | Equivalent token-wise recurrence using `jax.lax.scan` |
| FlashAttention and an FP32 output kernel | Plain masked attention with FP32 scores/output |
| Expert-parallel sparse dispatch | Evaluate all tiny experts, then mask unselected outputs |
| Distributed histogram Quantile Balancing | Exact quantile over one local batch |
| Incremental KDA/MLA caches | Full-sequence forward pass |
| BF16/MXFP4 distributed execution | Default JAX floating-point execution |
| MoonViT-V2 native vision tower | Omitted; this package implements the shared text backbone |
| Per-Head Muon and the full training recipe | Omitted; gradients work with any NNX/Optax optimizer |

The omitted systems are responsible for K3's practical trillion-parameter and
million-token efficiency. The mathematical structure here is useful for
learning and experiments, but it does not reproduce those scale claims.

## Source map

- `kimi_k3/config.py` - tiny defaults, paper dimensions, and the 3:1 layer schedule.
- `kimi_k3/model.py` - all model components and the causal language-model loss.
- `examples/tiny_forward.py` - forward and backward pass.
- `tests/test_model.py` - shapes, gradients, causality, decay bounds, and routing.

Primary references:

- Kimi Team, *Kimi K3: Open Frontier Intelligence*, especially §2 and Table 1
  (the PDF supplied with this task).
- Moonshot AI's
  [official configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
  and
  [official PyTorch implementation](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py).
