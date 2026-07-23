"""A small Sparse Mixture-of-Experts (MoE) using JAX and Flax NNX.

This example favors clarity over performance. Each input is routed to exactly
one expert, so only the selected expert processes that input.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class Expert(nnx.Module):
    """A simple two-layer neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear_1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.linear_1(x))
        return self.linear_2(x)


class SparseMoE(nnx.Module):
    """Route every input to one of several experts."""

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

        # The router produces one score for each expert.
        self.router = nnx.Linear(input_dim, num_experts, rngs=rngs)

        # nnx.List tells NNX that these experts belong to this model.
        self.experts = nnx.List(
            [
                Expert(input_dim, hidden_dim, output_dim, rngs=rngs)
                for _ in range(num_experts)
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Flatten leading dimensions so routing can handle either ordinary
        # batches [batch, features] or sequences [batch, length, features].
        leading_shape = x.shape[:-1]
        tokens = x.reshape((-1, self.input_dim))

        # Softmax converts router scores into expert probabilities.
        probabilities = jax.nn.softmax(self.router(tokens), axis=-1)

        # Top-1 routing chooses exactly one expert for each token.
        chosen_experts = jnp.argmax(probabilities, axis=-1)
        chosen_weights = jnp.take_along_axis(
            probabilities, chosen_experts[:, None], axis=-1
        )[:, 0]

        # Each branch calls one expert. lax.switch executes only the branch
        # selected by `expert_index`.
        expert_branches = tuple(
            lambda token, expert=expert: expert(token)
            for expert in self.experts
        )

        def run_one_token(inputs):
            token, expert_index, weight = inputs
            expert_output = jax.lax.switch(
                expert_index, expert_branches, token
            )

            # Multiplying by the router probability allows gradients to train
            # the router as well as the selected expert.
            return weight * expert_output

        # lax.map applies the sparse routing operation to every token.
        outputs = jax.lax.map(
            run_one_token,
            (tokens, chosen_experts, chosen_weights),
        )

        # Restore the original batch or sequence dimensions.
        return outputs.reshape((*leading_shape, self.output_dim))


if __name__ == "__main__":
    model = SparseMoE(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        num_experts=4,
        rngs=nnx.Rngs(0),
    )

    # Two example inputs, each of which may select a different expert.
    x = jnp.array(
        [
            [1.0, 0.0, 0.5, -1.0],
            [0.0, 1.0, -0.5, 1.0],
        ]
    )

    print(model(x))
