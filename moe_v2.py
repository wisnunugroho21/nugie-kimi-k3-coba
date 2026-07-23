"""A minimal routed Mixture of Experts (MoE) using JAX and Flax NNX.

Unlike ``moe.py``, this version does not evaluate every expert for every input.
The router selects one expert per input, and ``jax.lax.switch`` executes only
that expert's branch.

This is a teaching example. Processing inputs one at a time is easy to follow,
but production MoEs usually group inputs by expert for much better throughput.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class Expert(nnx.Module):
    """A small two-layer MLP used as one expert."""

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
        return self.output_layer(jax.nn.relu(self.input_layer(x)))


class RoutedMixtureOfExperts(nnx.Module):
    """Route each input to exactly one expert."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        *,
        rngs: nnx.Rngs,
    ):
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts

        # The router assigns a score to every expert.
        self.router = nnx.Linear(input_dim, num_experts, rngs=rngs)

        # nnx.List registers every expert and its trainable parameters.
        self.experts = nnx.List(
            [
                Expert(input_dim, hidden_dim, output_dim, rngs=rngs)
                for _ in range(num_experts)
            ]
        )

    def route(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the chosen expert indices and their probabilities."""
        probabilities = jax.nn.softmax(self.router(x), axis=-1)

        # Top-1 routing chooses exactly one expert for each input.
        expert_indices = jnp.argmax(probabilities, axis=-1)
        chosen_probabilities = jnp.take_along_axis(
            probabilities, expert_indices[..., None], axis=-1
        )[..., 0]
        return expert_indices, chosen_probabilities

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last input dimension {self.input_dim}, got {x.shape[-1]}"
            )

        original_shape = x.shape[:-1]
        flat_x = x.reshape((-1, self.input_dim))
        flat_indices, flat_probabilities = self.route(flat_x)

        # Each branch contains one expert. lax.switch runs only the branch
        # selected by expert_index for that input.
        expert_branches = tuple(
            lambda token, expert=expert: expert(token) for expert in self.experts
        )

        def run_selected_expert(inputs):
            token, expert_index, router_probability = inputs
            expert_output = jax.lax.switch(
                expert_index, expert_branches, token
            )

            # Weighting by the chosen router probability lets gradients train
            # the router. argmax itself is a discrete, non-differentiable choice.
            return router_probability * expert_output

        # lax.map preserves the per-input conditional. Using vmap over switch
        # can turn the conditional into a computation of every branch.
        flat_outputs = jax.lax.map(
            run_selected_expert,
            (flat_x, flat_indices, flat_probabilities),
        )

        return flat_outputs.reshape((*original_shape, self.output_dim))


if __name__ == "__main__":
    model = RoutedMixtureOfExperts(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        num_experts=4,
        rngs=nnx.Rngs(0),
    )

    # Each row is routed independently and may choose a different expert.
    x = jnp.array(
        [
            [1.0, 0.0, 0.5, -1.0],
            [0.0, 1.0, -0.5, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )

    expert_indices, probabilities = model.route(x)
    print("Selected experts:", expert_indices)
    print("Selected probabilities:", probabilities)
    print("MoE output:\n", model(x))
