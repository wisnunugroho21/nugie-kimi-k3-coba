# Kimi Linear with GDN-2, AttnRes, LatentMoE, and Gated MLA

This repository contains a compact decoder-only JAX/Flax NNX model. It keeps the
Kimi Linear 3:1 hybrid schedule, substitutes Gated DeltaNet-2 for Kimi Delta
Attention, and implements three additional architecture features:

- **Block Attention Residuals (AttnRes):** each token uses a learned pseudo-query
  to select normalized representations across depth. Queries start at zero, so
  depth weights are uniform at initialization. Full AttnRes and ordinary additive
  residuals remain available for experiments.
- **LatentMoE:** the routed path is projected from `d_model` to
  `moe_latent_dim`, dispatched and evaluated in that latent space, then projected
  back. Routing and shared experts remain at full width.
- **Gated Multi-head Latent Attention:** every full-attention layer applies a
  token- and head-specific sigmoid gate after latent attention and before the
  absorbed output projection. The gate adds no KV-cache state.

The default `KimiLinearConfig` enables all three features. Important controls are:

```python
KimiLinearConfig(
    attnres_mode="block",      # "block", "full", or "none"
    attnres_block_size=4,      # counts sublayers; decoder layer = 2 sublayers
    moe_latent_dim=64,         # None restores full-width routed experts
    moe_design_mode="accuracy", # "accuracy", "efficiency", or "custom"
    mla_gated=True,
)
```

### Optimized Attention Residuals

AttnRes uses a reusable source cache instead of stacking and RMS-normalizing the
entire depth history for every destination layer. Block mode also implements the
paper's exact two-phase schedule:

1. All pseudo-queries in the current block attend to completed block summaries
   in one batched operation.
2. Each sublayer merges its evolving intra-block partial through numerically
   stable online softmax.

The same residual execution engine is shared by training, prefill, and streaming
decode, preventing their block topology from drifting apart. Full AttnRes also
uses cached normalized sources, while retaining its original per-sublayer
semantics.

### LatentMoE design modes

`moe_n_routed` and `moe_top_k` describe the baseline standard-MoE values `N`
and `K`. With compression `alpha = d_model / moe_latent_dim`, the mode resolves
the experts actually instantiated:

| Mode | Routed experts | Active experts | Intended tradeoff |
| --- | ---: | ---: | --- |
| `efficiency` | `alpha * N` | `K` | Lower active expert cost |
| `accuracy` | `alpha * N` | `alpha * K` | Higher accuracy at comparable baseline cost |
| `custom` | `N` | `K` | Literal counts and legacy checkpoint compatibility |

`accuracy` is the default and the paper-recommended LatentMoE configuration.
Preset modes require an integer compression ratio. Inspect the resolved design
before constructing a large model:

```python
report = KimiLinearConfig().moe_design_report()
print(report["effective_n_routed"], report["effective_top_k"])
print(report["estimated_parameters_per_layer"])
print(report["estimated_flops_per_token_per_layer"])
```

## Padding-aware and packed-sequence training

`training.py` provides right-padding and packed-sequence collation, masked
next-token loss, AdamW with gradient clipping, MoE router-bias updates,
evaluation, and versioned model/optimizer checkpoints. Padding and document
boundaries are enforced inside the architecture: MLA cannot attend across
segments, GDN-2 resets both its recurrent state and short convolution at each
segment, and MoE load statistics count only real tokens.

```python
from flax import nnx

from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig
from training import (
    TrainingConfig,
    create_optimizer,
    make_lm_batch,
    make_packed_lm_batch,
    save_checkpoint,
    train_step,
)

model_config = KimiLinearConfig()
training_config = TrainingConfig()
model = KimiLinear(model_config, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, training_config)

# Raw token sequences include both the first input and final prediction target.
batch = make_lm_batch(
    [[1, 2, 3, 4], [5, 6]],
    pad_token_id=0,
    max_seq_len=model_config.max_seq_len,
)

# Or greedily combine documents into fixed-size rows. Targets are shifted before
# packing, so one document is never trained to predict the next document.
packed_batch = make_packed_lm_batch(
    [[1, 2, 3, 4], [5, 6], [7, 8, 9]],
    pad_token_id=0,
    max_seq_len=model_config.max_seq_len,
)
metrics = train_step(
    model,
    optimizer,
    batch,
    aux_loss_weight=training_config.aux_loss_weight,
    router_bias_lr=training_config.router_bias_lr,
)

save_checkpoint(
    "checkpoints/step_1",
    model,
    optimizer,
    model_config=model_config,
    training_config=training_config,
)
```

Use `train_epoch(...)` and `evaluate(...)` for iterables of pre-collated batches.
The model accepts integer or boolean `attention_mask` arrays with the same
`[batch, length]` shape as `input_ids`. Packed batches additionally carry integer
`segment_ids`; equal IDs identify tokens belonging to the same document within
each row, while padding uses `-1`.

Packed GDN-2 currently uses the exact token-recurrent scan so it can reset state
at arbitrary boundaries; unpacked batches retain the faster chunkwise training
core.

## Quick check

```bash
python3 -m pytest -q
```

For streaming inference, `model.step(...)` and `model.generate(...)` use the same
AttnRes depth topology as full-sequence training. AttnRes state exists only across
network depth during a forward pass; GDN-2 recurrent state and MLA latent KV state
remain the only time-axis caches.

## Architecture references

- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [LatentMoE](https://arxiv.org/abs/2601.18089)
- [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708)
- [Instella-MoE gated MLA reference](https://huggingface.co/amd/Instella-MoE-16B-A3B-Think)
