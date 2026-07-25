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
    mla_gated=True,
)
```

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
