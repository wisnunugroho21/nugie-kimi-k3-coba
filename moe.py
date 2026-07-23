"""A small, readable Mixture of Experts (MoE) layer using JAX and Flax NNX.

This example favors clarity over efficiency.  In particular, it evaluates every
expert for every input, then uses the router weights to mix their outputs.
Production MoE implementations usually dispatch each input only to its selected
experts to avoid this extra computation.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class Expert(nnx.Module):
    """A tiny two-layer MLP used as one expert."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_layer = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.output_layer = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Each expert learns a different nonlinear transformation of the input.
        return self.output_layer(jax.nn.relu(self.input_layer(x)))


class MixtureOfExperts(nnx.Module):
    """A minimal top-k Mixture of Experts layer.

    The final input dimension must equal ``input_dim``. Any leading dimensions
    are treated as batch dimensions, so both ``[batch, features]`` and
    ``[batch, sequence, features]`` inputs work.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        top_k: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.num_experts = num_experts
        self.top_k = top_k

        # The router produces one score (logit) per expert for every input.
        self.router = nnx.Linear(input_dim, num_experts, rngs=rngs)

        # nnx.List registers the experts and all of their trainable parameters.
        self.experts = nnx.List(
            [
                Expert(input_dim, hidden_dim, output_dim, rngs=rngs)
                for _ in range(num_experts)
            ]
        )

    def routing_weights(self, x: jax.Array) -> jax.Array:
        """Return normalized weights for the selected experts.

        The result has shape ``[..., num_experts]``. Unselected experts receive
        weight zero; selected expert weights add up to one for each input.
        """
        # Softmax turns arbitrary router scores into non-negative probabilities.
        probabilities = jax.nn.softmax(self.router(x), axis=-1)

        # Keep only the top-k probabilities. `top_k` returns both values and
        # integer expert indices, with the same leading dimensions as the input.
        _top_values, top_indices = jax.lax.top_k(probabilities, self.top_k)

        # Convert the selected indices into a multi-hot mask such as [1, 0, 1].
        # Summing is safe because top-k indices are always distinct.
        selected = jax.nn.one_hot(top_indices, self.num_experts).sum(axis=-2)
        sparse_weights = probabilities * selected

        # Removing experts also removes probability mass, so normalize again.
        # This makes the selected weights sum to one.
        return sparse_weights / sparse_weights.sum(axis=-1, keepdims=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        weights = self.routing_weights(x)

        # Shape: [num_experts, ..., output_dim].
        # Evaluating every expert is simple to understand, but not optimized.
        expert_outputs = jnp.stack([expert(x) for expert in self.experts])

        # Move the expert axis beside the output axis:
        # [num_experts, ..., output_dim] -> [..., num_experts, output_dim].
        expert_outputs = jnp.moveaxis(expert_outputs, 0, -2)

        # Weighted sum across experts, producing shape [..., output_dim].
        return jnp.sum(expert_outputs * weights[..., None], axis=-2)


if __name__ == "__main__":
    # NNX uses Rngs to initialize all trainable Linear parameters.
    model = MixtureOfExperts(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        num_experts=4,
        top_k=2,
        rngs=nnx.Rngs(0),
    )

    # Two example inputs. In a language model, these could be token embeddings.
    x = jnp.array(
        [
            [1.0, 0.0, 0.5, -1.0],
            [0.0, 1.0, -0.5, 1.0],
        ]
    )

    print("Routing weights:\n", model.routing_weights(x))
    print("MoE output:\n", model(x))
