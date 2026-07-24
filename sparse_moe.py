"""A small Sparse Mixture-of-Experts (MoE) using JAX and Flax NNX.

This example favors clarity over performance. Each input is routed to its top-k
experts, so only those selected experts process that input.
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
    """Route every input to its top-k experts."""

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
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.top_k = top_k

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

        # Top-k routing returns the k largest probabilities and their expert
        # indices. Shapes are [number_of_tokens, top_k].
        chosen_weights, chosen_experts = jax.lax.top_k(
            probabilities, self.top_k
        )

        # The selected probabilities no longer sum to one after the other
        # experts are removed, so normalize them again.
        chosen_weights = chosen_weights / chosen_weights.sum(
            axis=-1, keepdims=True
        )

        # Each branch calls one expert. lax.switch executes only the branch
        # selected by `expert_index`.
        expert_branches = tuple(
            lambda token, expert=expert: expert(token)
            for expert in self.experts
        )

        def run_one_expert(inputs):
            token, expert_index = inputs
            return jax.lax.switch(expert_index, expert_branches, token)

        # Repeat each token k times so each copy can visit one selected expert.
        repeated_tokens = jnp.repeat(tokens, self.top_k, axis=0)

        # lax.map executes only the selected expert for each token/expert pair.
        selected_outputs = jax.lax.map(
            run_one_expert,
            (repeated_tokens, chosen_experts.reshape(-1)),
        )

        # Group the k results for each token, multiply them by their normalized
        # router weights, and add them to form one output per token.
        selected_outputs = selected_outputs.reshape(
            (-1, self.top_k, self.output_dim)
        )
        outputs = jnp.sum(
            selected_outputs * chosen_weights[..., None], axis=1
        )

        # Restore the original batch or sequence dimensions.
        return outputs.reshape((*leading_shape, self.output_dim))


if __name__ == "__main__":
    model = SparseMoE(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        num_experts=4,
        top_k=2,
        rngs=nnx.Rngs(0),
    )

    # Each input is processed by its two highest-scoring experts.
    x = jnp.array(
        [
            [1.0, 0.0, 0.5, -1.0],
            [0.0, 1.0, -0.5, 1.0],
        ]
    )

    print(model(x))
