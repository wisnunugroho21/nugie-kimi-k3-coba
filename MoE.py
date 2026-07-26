"""Production-oriented LatentMoE in JAX and Flax NNX.

The paper's routed branch is:

    x -> down projection -> dispatch -> latent experts -> combine -> up projection

while the router and shared experts operate at the original model width.  This
file keeps that architecture and adds the systems pieces used by real MoEs:

* fixed-capacity expert buffers and configurable token dropping;
* fused grouped GEMMs through ``jax.lax.ragged_dot``;
* optional weight-and-activation INT8 expert inference;
* expert parallelism through ``nnx.shard_map`` and two all-to-all collectives.

The normal ``model(x)`` path works on one device.  ``expert_parallel_forward``
shards routed experts over a named device mesh and performs:

    dispatch all-to-all -> local expert GEMMs -> combine all-to-all.

Only the routed expert weights are sharded.  The router, latent projections, and
small shared-expert path are replicated, as in standard expert parallelism.

Paper: https://arxiv.org/abs/2601.18089
NVIDIA reference:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

Array = jax.Array
DropPolicy = Literal["position", "probability"]
Quantization = Literal["none", "int8"]


@dataclass(frozen=True)
class LatentMoEConfig:
    """Model and execution settings.

    The first five fields describe a baseline MoE.  LatentMoE derives its actual
    N' and K' using alpha = d_model / latent_dim:

    * both variants use N' = alpha * N routed experts;
    * the recommended "accuracy" variant uses K' = alpha * K;
    * the "efficiency" variant leaves K unchanged.

    ``capacity_factor`` multiplies the average assignments per expert.  A value
    above 1 reserves headroom for imperfect routing.  Overflow assignments are
    dropped according to ``drop_policy``.  ``capacity_multiple`` can align the
    buffer size for accelerator kernels.

    ``quantization="int8"`` stores expert weights as per-output-channel INT8 and
    dynamically quantizes expert inputs.  It is an inference configuration:
    only router/projection parameters remain trainable.
    """

    d_model: int
    latent_dim: int
    expert_hidden_dim: int
    baseline_num_experts: int
    baseline_top_k: int
    num_shared_experts: int = 0
    variant: Literal["accuracy", "efficiency"] = "accuracy"

    capacity_factor: float = 1.25
    min_capacity: int = 4
    capacity_multiple: int = 1
    drop_policy: DropPolicy = "probability"

    quantization: Quantization = "none"
    use_fused_grouped_gemm: bool = True

    def __post_init__(self) -> None:
        positive = {
            "d_model": self.d_model,
            "latent_dim": self.latent_dim,
            "expert_hidden_dim": self.expert_hidden_dim,
            "baseline_num_experts": self.baseline_num_experts,
            "baseline_top_k": self.baseline_top_k,
            "capacity_factor": self.capacity_factor,
            "min_capacity": self.min_capacity,
            "capacity_multiple": self.capacity_multiple,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.num_shared_experts < 0:
            raise ValueError("num_shared_experts cannot be negative.")
        if self.d_model % self.latent_dim != 0:
            raise ValueError("d_model must be divisible by latent_dim.")
        if self.baseline_top_k > self.baseline_num_experts:
            raise ValueError("baseline_top_k cannot exceed baseline_num_experts.")
        if self.variant not in ("accuracy", "efficiency"):
            raise ValueError("variant must be 'accuracy' or 'efficiency'.")
        if self.drop_policy not in ("position", "probability"):
            raise ValueError("drop_policy must be 'position' or 'probability'.")
        if self.quantization not in ("none", "int8"):
            raise ValueError("quantization must be 'none' or 'int8'.")

    @property
    def compression_ratio(self) -> int:
        """alpha = d / ell."""
        return self.d_model // self.latent_dim

    @property
    def num_routed_experts(self) -> int:
        """N' = alpha * N."""
        return self.compression_ratio * self.baseline_num_experts

    @property
    def top_k(self) -> int:
        """K' = alpha * K for accuracy, or K for efficiency."""
        multiplier = self.compression_ratio if self.variant == "accuracy" else 1
        return multiplier * self.baseline_top_k


class RoutingInfo(NamedTuple):
    """Router diagnostics returned without mutating module state."""

    probabilities: Array  # [..., N']
    expert_indices: Array  # [..., K']
    expert_weights: Array  # [..., K']
    kept_mask: Array  # [..., K']; False means dropped by capacity
    expert_capacity: Array  # scalar; total slots per expert
    dropped_fraction: Array  # scalar in [0, 1]
    load_balancing_loss: Array  # scalar, 1 at perfectly uniform routing


class Int8Weight(nnx.Variable):
    """Non-trainable INT8 expert weight stored in NNX state/checkpoints."""


class QuantScale(nnx.Variable):
    """Non-trainable scale associated with an ``Int8Weight``."""


class DispatchPlan(NamedTuple):
    """Static-capacity routing metadata for sending results back to tokens."""

    buffer: Array  # [E, C, D]
    flat_token_indices: Array  # [T*K]
    flat_expert_indices: Array  # [T*K]
    slot_indices: Array  # [T*K]
    flat_weights: Array  # [T*K]
    kept_mask: Array  # [T*K]
    capacity: int  # Python integer: shapes must be static under JIT


def _xavier_uniform(key: Array, shape: tuple[int, int, int]) -> Array:
    """Xavier initialization for [expert, input, output] weight banks."""
    _, fan_in, fan_out = shape
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def _quantize_weight(kernel: Array) -> tuple[Array, Array]:
    """Symmetric per-expert, per-output-channel weight quantization.

    ``kernel`` is [E, input, output].  The scale is [E, 1, output], so each
    output channel gets its own range and can be restored after INT8 GEMM.
    """
    scale = jnp.max(jnp.abs(kernel), axis=1, keepdims=True) / 127.0
    scale = jnp.maximum(scale, jnp.finfo(jnp.float32).tiny)
    quantized = jnp.clip(jnp.round(kernel / scale), -127, 127).astype(jnp.int8)
    return quantized, scale.astype(jnp.float32)


def _dynamic_quantize_activation(x: Array) -> tuple[Array, Array]:
    """Per-row activation quantization for x shaped [E, C, input]."""
    scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / 127.0
    scale = jnp.maximum(scale, jnp.finfo(jnp.float32).tiny)
    quantized = jnp.clip(jnp.round(x / scale), -127, 127).astype(jnp.int8)
    return quantized, scale.astype(jnp.float32)


def _capacity(
    num_assignments: int,
    num_experts: int,
    factor: float,
    minimum: int,
    multiple: int,
) -> int:
    """Static expert capacity, rounded up for kernel-friendly alignment."""
    raw = max(minimum, math.ceil(factor * num_assignments / num_experts))
    return math.ceil(raw / multiple) * multiple


def _make_dispatch_plan(
    tokens: Array,
    expert_indices: Array,
    expert_weights: Array,
    *,
    num_experts: int,
    capacity_factor: float,
    min_capacity: int,
    capacity_multiple: int,
    drop_policy: DropPolicy,
) -> DispatchPlan:
    """Pack top-k assignments into a fixed [expert, capacity, width] buffer.

    Sorting makes each expert's assignments contiguous.  The rank inside that
    group becomes the capacity slot.  Probability dropping keeps the strongest
    routes; position dropping keeps the earliest routes and is cheaper to sort.
    """
    num_tokens, top_k = expert_indices.shape
    num_assignments = num_tokens * top_k
    capacity = _capacity(
        num_assignments,
        num_experts,
        capacity_factor,
        min_capacity,
        capacity_multiple,
    )

    flat_experts = expert_indices.reshape(-1)
    flat_weights = expert_weights.reshape(-1)
    flat_tokens = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
    assignment_order = jnp.arange(num_assignments, dtype=jnp.int32)

    # jnp.lexsort uses the final key as primary.  The assignment index provides
    # deterministic tie-breaking, which is valuable for reproducible training.
    if drop_policy == "probability":
        order = jnp.lexsort((assignment_order, -flat_weights, flat_experts))
    else:
        order = jnp.lexsort((assignment_order, flat_experts))

    sorted_experts = flat_experts[order]
    is_group_start = jnp.concatenate(
        [jnp.ones((1,), dtype=bool), sorted_experts[1:] != sorted_experts[:-1]]
    )
    positions = jnp.arange(num_assignments, dtype=jnp.int32)
    starts = jnp.where(is_group_start, positions, 0)
    group_starts = lax.associative_scan(jnp.maximum, starts)
    sorted_slots = positions - group_starts

    # Undo the sorting permutation so every original token/top-k assignment has
    # its expert-local slot number.
    slots = jnp.zeros_like(sorted_slots).at[order].set(sorted_slots)
    kept = slots < capacity
    safe_slots = jnp.minimum(slots, capacity - 1)

    # Dropped assignments contribute zeros.  Kept assignments have unique
    # (expert, slot) addresses, so scatter-add is deterministic.
    updates = tokens[flat_tokens] * kept[:, None]
    buffer = jnp.zeros((num_experts, capacity, tokens.shape[-1]), dtype=tokens.dtype)
    buffer = buffer.at[flat_experts, safe_slots].add(updates)

    return DispatchPlan(
        buffer,
        flat_tokens,
        flat_experts,
        safe_slots,
        flat_weights,
        kept,
        capacity,
    )


def _combine_dispatch(
    plan: DispatchPlan, expert_output: Array, num_tokens: int
) -> Array:
    """Apply router weights and scatter expert outputs back to their tokens."""
    selected = expert_output[plan.flat_expert_indices, plan.slot_indices]
    selected = selected * plan.kept_mask[:, None]
    weighted = selected * plan.flat_weights[:, None]
    output = jnp.zeros((num_tokens, expert_output.shape[-1]), expert_output.dtype)
    return output.at[plan.flat_token_indices].add(weighted)


class ExpertBank(nnx.Module):
    """SwiGLU experts evaluated with grouped GEMMs.

    FC1 and gate are concatenated into one [D, 2M] matrix, matching the common
    fused SwiGLU implementation.  ``lax.ragged_dot`` consumes contiguous expert
    groups in one primitive instead of launching one matmul per expert.
    """

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        *,
        quantization: Quantization,
        use_fused_grouped_gemm: bool,
        rngs: nnx.Rngs,
    ):
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.quantization = quantization
        self.use_fused_grouped_gemm = use_fused_grouped_gemm

        gate_up = _xavier_uniform(
            rngs.params(), (num_experts, input_dim, 2 * hidden_dim)
        )
        down = _xavier_uniform(rngs.params(), (num_experts, hidden_dim, input_dim))

        if quantization == "none":
            self.gate_up_kernel = nnx.Param(gate_up)
            self.down_kernel = nnx.Param(down)
            self.gate_up_scale = None
            self.down_scale = None
        else:
            gate_up_q, gate_up_scale = _quantize_weight(gate_up)
            down_q, down_scale = _quantize_weight(down)
            self.gate_up_kernel = Int8Weight(gate_up_q)
            self.down_kernel = Int8Weight(down_q)
            self.gate_up_scale = QuantScale(gate_up_scale)
            self.down_scale = QuantScale(down_scale)

    def _grouped_dot(
        self,
        x: Array,
        kernel: Array,
        scale: Array | None,
    ) -> Array:
        """[E,C,K] @ [E,K,N] -> [E,C,N], using one grouped primitive."""
        num_experts, capacity, _ = x.shape

        if self.quantization == "int8":
            assert scale is not None
            qx, x_scale = _dynamic_quantize_activation(x)
            lhs = qx.reshape(num_experts * capacity, -1)
            preferred_type = jnp.int32
        else:
            lhs = x.reshape(num_experts * capacity, -1)
            x_scale = None
            preferred_type = jnp.float32

        if self.use_fused_grouped_gemm:
            # All groups have the same padded capacity.  On accelerator backends
            # ragged_dot lowers as a grouped matrix multiplication.
            group_sizes = jnp.full((num_experts,), capacity, dtype=jnp.int32)
            result = lax.ragged_dot(
                lhs,
                kernel,
                group_sizes,
                preferred_element_type=preferred_type,
            ).reshape(num_experts, capacity, -1)
        else:
            # Portable correctness fallback: a batched dot with expert as the
            # batch dimension.  It is also useful for backend comparisons.
            result = lax.dot_general(
                x if self.quantization == "none" else qx,
                kernel,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=preferred_type,
            )

        if self.quantization == "int8":
            # int32 accumulator * activation scale * weight scale -> fp32.
            result = result.astype(jnp.float32) * x_scale * scale
        return result

    def dispatched(self, x: Array) -> Array:
        """Evaluate buffers shaped [local_experts, capacity, input_dim]."""
        gate_up = self._grouped_dot(
            x,
            self.gate_up_kernel[...],
            None if self.gate_up_scale is None else self.gate_up_scale[...],
        )
        gate, up = jnp.split(gate_up, 2, axis=-1)
        hidden = jax.nn.silu(gate) * up
        return self._grouped_dot(
            hidden,
            self.down_kernel[...],
            None if self.down_scale is None else self.down_scale[...],
        )

    def all(self, x: Array) -> Array:
        """Run every shared expert for every token and sum their outputs."""
        dispatched = jnp.broadcast_to(
            x[None, :, :], (self.num_experts, x.shape[0], self.input_dim)
        )
        return self.dispatched(dispatched).sum(axis=0)


class LatentMoE(nnx.Module):
    """Capacity-aware LatentMoE layer.

    ``expert_parallel_axis`` is normally left as None.  The public
    ``expert_parallel_forward`` wrapper supplies it inside ``nnx.shard_map``,
    where the routed expert parameters have a smaller local expert dimension.
    """

    def __init__(self, config: LatentMoEConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.router = nnx.Linear(
            config.d_model,
            config.num_routed_experts,
            use_bias=False,
            rngs=rngs,
        )
        self.down_projection = nnx.Linear(
            config.d_model, config.latent_dim, use_bias=False, rngs=rngs
        )
        self.up_projection = nnx.Linear(
            config.latent_dim, config.d_model, use_bias=False, rngs=rngs
        )
        self.routed_experts = ExpertBank(
            config.num_routed_experts,
            config.latent_dim,
            config.expert_hidden_dim,
            quantization=config.quantization,
            use_fused_grouped_gemm=config.use_fused_grouped_gemm,
            rngs=rngs,
        )
        self.shared_experts = (
            ExpertBank(
                config.num_shared_experts,
                config.d_model,
                config.expert_hidden_dim,
                quantization=config.quantization,
                use_fused_grouped_gemm=config.use_fused_grouped_gemm,
                rngs=rngs,
            )
            if config.num_shared_experts
            else None
        )

    def _local_experts(
        self,
        latent_tokens: Array,
        expert_indices: Array,
        expert_weights: Array,
    ) -> tuple[Array, Array, int]:
        """Single-device capacity dispatch -> grouped GEMM -> combine."""
        plan = _make_dispatch_plan(
            latent_tokens,
            expert_indices,
            expert_weights,
            num_experts=self.config.num_routed_experts,
            capacity_factor=self.config.capacity_factor,
            min_capacity=self.config.min_capacity,
            capacity_multiple=self.config.capacity_multiple,
            drop_policy=self.config.drop_policy,
        )
        expert_output = self.routed_experts.dispatched(plan.buffer)
        mixed = _combine_dispatch(plan, expert_output, latent_tokens.shape[0])
        return mixed, plan.kept_mask, plan.capacity

    def _parallel_experts(
        self,
        latent_tokens: Array,
        expert_indices: Array,
        expert_weights: Array,
        axis_name: str,
    ) -> tuple[Array, Array, int]:
        """All-to-all expert dispatch, local grouped GEMMs, and reverse exchange.

        Each source reserves C slots for every global expert.  After all-to-all,
        an expert receives C slots from every source, for total capacity EP*C.
        """
        global_experts = self.config.num_routed_experts

        # Under StateSharding, the actual leading array dimension is local even
        # though the module's static config still records the global expert count.
        local_experts = self.routed_experts.gate_up_kernel[...].shape[0]
        if global_experts % local_experts != 0:
            raise ValueError("Routed experts must shard evenly over the EP mesh.")
        ep_size = global_experts // local_experts

        plan = _make_dispatch_plan(
            latent_tokens,
            expert_indices,
            expert_weights,
            num_experts=global_experts,
            capacity_factor=self.config.capacity_factor,
            min_capacity=self.config.min_capacity,
            capacity_multiple=self.config.capacity_multiple,
            drop_policy=self.config.drop_policy,
        )

        # Global expert ID is laid out as:
        #   destination_rank * local_experts + local_expert_id.
        send = plan.buffer.reshape(
            ep_size,
            local_experts,
            plan.capacity,
            self.config.latent_dim,
        )

        # Matrix transpose across source and destination ranks.  On each
        # destination the new leading axis enumerates source ranks.
        received = lax.all_to_all(
            send, axis_name, split_axis=0, concat_axis=0, tiled=False
        )

        # Group all source slots belonging to the same local expert so a single
        # grouped-GEMM launch processes the entire received buffer.
        local_input = received.transpose(1, 0, 2, 3).reshape(
            local_experts,
            ep_size * plan.capacity,
            self.config.latent_dim,
        )
        local_output = self.routed_experts.dispatched(local_input)

        # Restore [source, local_expert, C, ell], then transpose the device
        # communication matrix again to return outputs to their source tokens.
        send_back = local_output.reshape(
            local_experts,
            ep_size,
            plan.capacity,
            self.config.latent_dim,
        ).transpose(1, 0, 2, 3)
        returned = lax.all_to_all(
            send_back, axis_name, split_axis=0, concat_axis=0, tiled=False
        )
        returned = returned.reshape(
            global_experts, plan.capacity, self.config.latent_dim
        )

        mixed = _combine_dispatch(plan, returned, latent_tokens.shape[0])
        return mixed, plan.kept_mask, ep_size * plan.capacity

    def __call__(
        self,
        x: Array,
        *,
        expert_parallel_axis: str | None = None,
    ) -> tuple[Array, RoutingInfo]:
        """Apply LatentMoE to ``[..., d_model]`` tokens."""
        if x.ndim < 2 or x.shape[-1] != self.config.d_model:
            raise ValueError(
                f"x must have shape [..., {self.config.d_model}], got {x.shape}."
            )
        leading_shape = x.shape[:-1]
        if math.prod(leading_shape) == 0:
            raise ValueError("LatentMoE requires at least one token.")
        tokens = x.reshape(-1, self.config.d_model)

        # The paper routes from full-width x, before the latent down projection.
        logits = self.router(tokens).astype(jnp.float32)
        probabilities = jax.nn.softmax(logits, axis=-1)
        expert_weights, expert_indices = lax.top_k(probabilities, self.config.top_k)

        latent_tokens = self.down_projection(tokens)
        if expert_parallel_axis is None:
            mixed_latents, kept, capacity = self._local_experts(
                latent_tokens, expert_indices, expert_weights
            )
        else:
            mixed_latents, kept, capacity = self._parallel_experts(
                latent_tokens,
                expert_indices,
                expert_weights,
                expert_parallel_axis,
            )

        output = self.up_projection(mixed_latents)
        if self.shared_experts is not None:
            output = output + self.shared_experts.all(tokens)

        # Load balance is measured before capacity dropping, as is customary.
        assignments = jax.nn.one_hot(expert_indices, self.config.num_routed_experts)
        assignment_fraction = assignments.mean(axis=(0, 1))
        mean_probability = probabilities.mean(axis=0)
        dropped_fraction = 1.0 - kept.astype(jnp.float32).mean()

        # In EP mode each shard sees a different token subset.  Make router
        # statistics global and replicated on every rank.
        if expert_parallel_axis is not None:
            assignment_fraction = lax.pmean(assignment_fraction, expert_parallel_axis)
            mean_probability = lax.pmean(mean_probability, expert_parallel_axis)
            dropped_fraction = lax.pmean(dropped_fraction, expert_parallel_axis)

        load_balancing_loss = self.config.num_routed_experts * jnp.sum(
            assignment_fraction * mean_probability
        )

        info = RoutingInfo(
            probabilities.reshape(*leading_shape, self.config.num_routed_experts),
            expert_indices.reshape(*leading_shape, self.config.top_k),
            expert_weights.reshape(*leading_shape, self.config.top_k),
            kept.reshape(*leading_shape, self.config.top_k),
            jnp.asarray(capacity, jnp.int32),
            dropped_fraction,
            load_balancing_loss,
        )
        return output.reshape(*leading_shape, self.config.d_model), info


def quantize_for_inference(model: LatentMoE) -> LatentMoE:
    """Convert a trained float model into a weight-only INT8 checkpoint.

    Expert activations are then quantized dynamically on every call.  Router and
    latent projections stay in float, which avoids quantizing the sensitive
    routing decision and mirrors common mixed-precision MoE deployments.
    """
    if model.config.quantization != "none":
        raise ValueError("quantize_for_inference expects a float source model.")

    quantized = LatentMoE(
        replace(model.config, quantization="int8"),
        # Initialization is overwritten below; this key only constructs shapes.
        rngs=nnx.Rngs(0),
    )

    # Replicated, trainable parts remain in their original floating-point dtype.
    quantized.router.kernel[...] = model.router.kernel[...]
    quantized.down_projection.kernel[...] = model.down_projection.kernel[...]
    quantized.up_projection.kernel[...] = model.up_projection.kernel[...]

    def copy_expert_bank(source: ExpertBank, target: ExpertBank) -> None:
        gate_up_q, gate_up_scale = _quantize_weight(source.gate_up_kernel[...])
        down_q, down_scale = _quantize_weight(source.down_kernel[...])
        target.gate_up_kernel[...] = gate_up_q
        target.down_kernel[...] = down_q
        assert target.gate_up_scale is not None and target.down_scale is not None
        target.gate_up_scale[...] = gate_up_scale
        target.down_scale[...] = down_scale

    copy_expert_bank(model.routed_experts, quantized.routed_experts)
    if model.shared_experts is not None:
        assert quantized.shared_experts is not None
        copy_expert_bank(model.shared_experts, quantized.shared_experts)
    return quantized


def _path_ends_with_routed_expert_weight(path, _value) -> bool:
    """NNX state filter selecting only arrays in the routed ExpertBank."""
    return (
        len(path) >= 2
        and path[-2] == "routed_experts"
        and path[-1] in {"gate_up_kernel", "down_kernel", "gate_up_scale", "down_scale"}
    )


def expert_parallel_forward(
    model: LatentMoE,
    x: Array,
    mesh: Mesh,
    *,
    axis_name: str = "expert",
) -> tuple[Array, RoutingInfo]:
    """Run LatentMoE with routed experts sharded across ``mesh``.

    Args:
        model: A normally initialized global ``LatentMoE``.
        x: Tokens shaped [T, d_model]. T must divide evenly over the EP axis.
        mesh: One- or multi-dimensional JAX mesh containing ``axis_name``.
        axis_name: Mesh axis used for expert parallelism.

    This wrapper intentionally requires a flat token dimension.  In a full model,
    the surrounding data/sequence parallel layout should flatten its local tokens
    before this call and restore batch/sequence dimensions afterward.
    """
    if x.ndim != 2 or x.shape[-1] != model.config.d_model:
        raise ValueError(
            f"expert_parallel_forward expects [T, {model.config.d_model}], "
            f"got {x.shape}."
        )
    if axis_name not in mesh.axis_names:
        raise ValueError(f"Mesh has no axis named {axis_name!r}.")

    ep_size = mesh.shape[axis_name]
    if x.shape[0] % ep_size:
        raise ValueError("The token count must be divisible by the EP mesh size.")
    if model.config.num_routed_experts % ep_size:
        raise ValueError("The routed expert count must divide over the EP mesh.")

    # Shard only routed-expert arrays on their leading expert dimension.
    # Router, projections, and shared experts match the catch-all replicated spec.
    model_spec = nnx.StateSharding(
        [
            (_path_ends_with_routed_expert_weight, P(axis_name, None, None)),
            (..., P()),
        ]
    )
    info_spec = RoutingInfo(
        P(axis_name, None),
        P(axis_name, None),
        P(axis_name, None),
        P(axis_name, None),
        P(),
        P(),
        P(),
    )

    @nnx.shard_map(
        mesh=mesh,
        in_specs=(model_spec, P(axis_name, None)),
        out_specs=(P(axis_name, None), info_spec),
        axis_names={axis_name},
    )
    def parallel_call(m: LatentMoE, local_x: Array):
        return m(local_x, expert_parallel_axis=axis_name)

    return parallel_call(model, x)


if __name__ == "__main__":
    config = LatentMoEConfig(
        d_model=64,
        latent_dim=16,  # alpha = 4
        expert_hidden_dim=32,
        baseline_num_experts=4,
        baseline_top_k=1,
        num_shared_experts=1,
        variant="accuracy",  # N'=16 and K'=4
        capacity_factor=1.25,
        quantization="none",  # use "int8" for inference
    )
    model = LatentMoE(config, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 8, config.d_model))
    y, routing = model(x)

    print("input / output:", x.shape, y.shape)
    print("routed / active experts:", config.num_routed_experts, config.top_k)
    print("capacity per expert:", int(routing.expert_capacity))
    print("dropped assignments:", float(routing.dropped_fraction))
    print("load-balancing loss:", float(routing.load_balancing_loss))
