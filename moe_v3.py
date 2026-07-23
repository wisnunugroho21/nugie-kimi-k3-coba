"""A high-throughput, single-device Mixture of Experts using JAX and Flax NNX.

The important performance ideas are:

1. Route each token to one expert (top-1 routing).
2. Pack tokens into fixed-size expert batches with one scatter operation.
3. Store all expert weights in stacked arrays.
4. Evaluate all expert batches with two large batched matrix multiplications.
5. Keep every shape static so the whole forward pass works well with ``nnx.jit``.

This is much more accelerator-friendly than running a Python loop per token.
It is still intentionally compact: a distributed MoE would additionally shard
experts across devices and exchange tokens with an all-to-all collective.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx


class FastMixtureOfExperts(nnx.Module):
    """A grouped, top-1 MoE layer with a fixed capacity per expert."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        *,
        rngs: nnx.Rngs,
    ):
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # A regular NNX Linear layer is sufficient for the small router.
        self.router = nnx.Linear(input_dim, num_experts, rngs=rngs)

        # Expert parameters are stacked on their first axis. This lets XLA use
        # batched matrix multiplications instead of many small Python-level MLPs.
        initialize = nnx.initializers.lecun_normal()
        keys_1 = jax.random.split(rngs.params(), num_experts)
        keys_2 = jax.random.split(rngs.params(), num_experts)

        self.kernel_1 = nnx.Param(
            jax.vmap(lambda key: initialize(key, (input_dim, hidden_dim)))(keys_1)
        )
        self.bias_1 = nnx.Param(jnp.zeros((num_experts, hidden_dim)))
        self.kernel_2 = nnx.Param(
            jax.vmap(lambda key: initialize(key, (hidden_dim, output_dim)))(keys_2)
        )
        self.bias_2 = nnx.Param(jnp.zeros((num_experts, output_dim)))

    def _route(
        self, tokens: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return probabilities, chosen expert indices, and chosen gate values."""
        probabilities = jax.nn.softmax(self.router(tokens), axis=-1)
        gates, expert_indices = jax.lax.top_k(probabilities, 1)
        return probabilities, expert_indices[:, 0], gates[:, 0]

    def forward_with_aux(
        self, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(output, load_balancing_loss)``.

        The auxiliary loss encourages the router to use all experts. A training
        objective can add a small multiple of it, for example:

        ``total_loss = task_loss + 0.01 * load_balancing_loss``
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last input dimension {self.input_dim}, got {x.shape[-1]}"
            )

        original_shape = x.shape[:-1]
        tokens = x.reshape((-1, self.input_dim))
        num_tokens = tokens.shape[0]
        if num_tokens == 0:
            raise ValueError("The input must contain at least one token")

        probabilities, expert_indices, gates = self._route(tokens)

        # Capacity is static at compilation time. A factor above 1 gives the
        # router some imbalance tolerance while keeping memory bounded.
        capacity = max(
            1,
            math.ceil(
                self.capacity_factor * num_tokens / self.num_experts
            ),
        )

        # assignments[t, e] is 1 when token t selected expert e.
        assignments = jax.nn.one_hot(
            expert_indices, self.num_experts, dtype=jnp.int32
        )

        # Cumulative counts give each token a unique slot inside its expert.
        positions_for_all_experts = jnp.cumsum(assignments, axis=0) - 1
        positions = jnp.sum(
            positions_for_all_experts * assignments, axis=-1
        )

        # Tokens beyond an expert's capacity are dropped. Clipping gives every
        # scatter a valid index; multiplying by `kept` makes dropped updates zero.
        kept = positions < capacity
        safe_positions = jnp.minimum(positions, capacity - 1)

        # Pack [tokens, input_dim] into [experts, capacity, input_dim].
        # Array.at lowers to an accelerator scatter operation.
        expert_inputs = jnp.zeros(
            (self.num_experts, capacity, self.input_dim),
            dtype=tokens.dtype,
        )
        expert_inputs = expert_inputs.at[
            expert_indices, safe_positions
        ].add(tokens * kept[:, None])

        # Run every expert as part of two batched matrix multiplications.
        # Only the compact expert buffers are processed, not every token by
        # every expert as in a dense MoE.
        hidden = jnp.einsum(
            "ecd,edh->ech", expert_inputs, self.kernel_1[...]
        )
        hidden = jax.nn.gelu(hidden + self.bias_1[:, None, :])
        expert_outputs = jnp.einsum(
            "ech,eho->eco", hidden, self.kernel_2[...]
        )
        expert_outputs = expert_outputs + self.bias_2[:, None, :]

        # Gather each token's result from the expert buffer and restore the
        # original leading dimensions. The gate trains the router; dropped
        # tokens become zeros and therefore do not affect downstream values.
        outputs = expert_outputs[expert_indices, safe_positions]
        outputs = outputs * gates[:, None] * kept[:, None]
        outputs = outputs.reshape((*original_shape, self.output_dim))

        # This standard auxiliary term is smallest when traffic and average
        # router probabilities are both evenly distributed across experts.
        traffic_fraction = assignments.mean(axis=0)
        probability_fraction = probabilities.mean(axis=0)
        load_balancing_loss = self.num_experts * jnp.sum(
            traffic_fraction * probability_fraction
        )

        return outputs, load_balancing_loss

    def __call__(self, x: jax.Array) -> jax.Array:
        output, _load_balancing_loss = self.forward_with_aux(x)
        return output


if __name__ == "__main__":
    model = FastMixtureOfExperts(
        input_dim=128,
        hidden_dim=256,
        output_dim=128,
        num_experts=8,
        capacity_factor=1.25,
        rngs=nnx.Rngs(0),
    )

    # In a transformer this shape could mean [batch, sequence, model_dimension].
    x = jax.random.normal(jax.random.key(1), (4, 32, 128))

    # JIT compilation is essential for the scatter, gather, and batched matrix
    # multiplications to be fused/optimized by XLA.
    fast_forward = nnx.jit(lambda module, inputs: module.forward_with_aux(inputs))
    output, auxiliary_loss = fast_forward(model, x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Load-balancing loss:", float(auxiliary_loss))
