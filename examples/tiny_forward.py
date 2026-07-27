"""Construct the miniature K3 and run a forward/backward pass."""

import jax
import jax.numpy as jnp
from flax import nnx

from kimi_k3 import KimiK3, KimiK3Config, causal_lm_loss


def main() -> None:
    """Print model shapes and prove that the implementation is differentiable."""
    config = KimiK3Config.tiny(vocab_size=128, hidden_size=32, head_dim=8)
    model = KimiK3(config, rngs=nnx.Rngs(0))
    tokens = jax.random.randint(jax.random.key(1), (2, 12), 0, config.vocab_size)

    logits = model(tokens)
    loss = causal_lm_loss(logits, tokens)
    gradients = nnx.grad(lambda m: causal_lm_loss(m(tokens), tokens))(model)

    parameter_count = sum(
        value.size for value in jax.tree.leaves(nnx.state(model, nnx.Param))
    )
    gradient_count = sum(value.size for value in jax.tree.leaves(gradients))
    print(f"logits: {logits.shape}")
    print(f"loss: {float(loss):.4f}")
    print(f"parameters: {parameter_count:,}")
    print(f"gradient entries: {gradient_count:,}")


if __name__ == "__main__":
    main()
