"""Equinox implementation of a Neural Central Pattern Generator (CPG) model."""

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.typing import ArrayLike

from neural_actor import AbstractNeuralActor, AbstractOutputMapping, AbstractVectorField
from utils import mlp_init


def split_states(states: ArrayLike, num_oscillators: int) -> tuple[Array, Array, Array]:
    states = jnp.asarray(states)

    amplitudes = states[:num_oscillators]
    phases = states[num_oscillators : num_oscillators * 2]
    amplitudes_dot = states[num_oscillators * 2 :]

    return amplitudes, phases, amplitudes_dot


def merge_states(
    amplitudes: ArrayLike, phases: ArrayLike, amplitudes_dot: ArrayLike
) -> Array:
    return jnp.concatenate([amplitudes, phases, amplitudes_dot])


def split_params(
    params: ArrayLike, num_oscillators: int
) -> tuple[Array, Array, Array, Array]:
    params = jnp.asarray(params)

    intrinsic_amplitudes = params[:num_oscillators]
    intrinsic_frequencies = params[num_oscillators : num_oscillators * 2]
    coupling_weights = params[
        num_oscillators * 2 : num_oscillators * 2 + num_oscillators**2
    ].reshape(num_oscillators, num_oscillators)
    phase_biases = params[num_oscillators * 2 + num_oscillators**2 :].reshape(
        num_oscillators, num_oscillators
    )

    return intrinsic_amplitudes, intrinsic_frequencies, coupling_weights, phase_biases


def merge_params(
    intrinsic_amplitudes: ArrayLike,
    intrinsic_frequencies: ArrayLike,
    coupling_weights: ArrayLike,
    phase_biases: ArrayLike,
) -> Array:
    coupling_weights = jnp.asarray(coupling_weights).flatten()
    phase_biases = jnp.asarray(phase_biases).flatten()
    return jnp.concatenate(
        [intrinsic_amplitudes, intrinsic_frequencies, coupling_weights, phase_biases]
    )


def cpg_vector_field(
    num_oscillators: int,
    convergence_factor: float,
    _: float,
    state: ArrayLike,
    params: ArrayLike,
) -> Array:
    """Central Pattern Generator (CPG) vector field.

    Based on the amplitude controlled phase oscillator model described in:
    @misc{bellegarda2022cpgrllearningcentralpattern,
        title={CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion},
        author={Guillaume Bellegarda and Auke Ijspeert},
        year={2022},
        eprint={2211.00458},
        archivePrefix={arXiv},
        primaryClass={cs.RO},
        url={https://arxiv.org/abs/2211.00458},
    }
    """
    assert jnp.asarray(state).shape[0] == 3 * num_oscillators, "Invalid state shape"
    assert jnp.asarray(params).shape[0] == 2 * (
        num_oscillators + num_oscillators**2
    ), "Invalid params shape"

    amplitudes, phases, amplitudes_dot = split_states(state, num_oscillators)
    intrinsic_amplitudes, intrinsic_frequencies, coupling_weights, phase_biases = (
        split_params(params, num_oscillators)
    )

    phase_diffs = jnp.subtract.outer(phases, phases) - phase_biases
    coupling_terms = amplitudes.reshape(-1, 1) * coupling_weights * jnp.sin(phase_diffs)
    phase_dot = intrinsic_frequencies + coupling_terms.sum(axis=1)

    amplitudes_dot_dot = convergence_factor * (
        (convergence_factor / 4) * (intrinsic_amplitudes - amplitudes) - amplitudes_dot
    )

    return jnp.concatenate([amplitudes_dot, phase_dot, amplitudes_dot_dot])


def cpg_output(state: ArrayLike, num_oscillators: int) -> Array:
    amplitudes, phases, _ = split_states(state, num_oscillators)
    return jnp.concatenate([amplitudes * jnp.cos(phases), amplitudes * jnp.sin(phases)])


class ForcedCPG(AbstractVectorField, strict=True):
    """Parameterized CPG vector field with an external input."""

    num_oscillators: int
    convergence_factor: float
    input_mapping: eqx.nn.MLP
    state_shape: int
    param_shape: int

    def __init__(
        self,
        num_oscillators: int,
        convergence_factor: float,
        input_size: int,
        input_mapping_width: int,
        input_mapping_depth: int,
        *,
        key: Array,
    ) -> None:
        self.num_oscillators = num_oscillators
        self.convergence_factor = convergence_factor  # this is trainable
        self.state_shape = 3 * num_oscillators
        self.param_shape = 2 * (num_oscillators + num_oscillators**2)

        # Using a small std for the layers can make the early stages of training more stable
        self.input_mapping = mlp_init(
            in_size=input_size + 1 + self.state_shape,
            out_size=self.param_shape,
            width_size=input_mapping_width,
            depth=input_mapping_depth,
            activation=jax.nn.softplus,  # Continuously differentiable activation function theoretically required
            final_activation=jax.nn.tanh,
            hidden_std=0.01,
            final_std=0.1,
            key=key,
        )

    def __call__(self, t: float, y: Array, x: Array) -> Array:
        params = self.input_mapping(jnp.concatenate([x, jnp.asarray([t]), y]))
        return cpg_vector_field(
            self.num_oscillators, self.convergence_factor, t, y, params
        )


class CPGOutputMap(AbstractOutputMapping, strict=True):
    """Map a CPG state to an output."""

    num_oscillators: int
    output_mapping: eqx.nn.MLP
    output_shape: int

    def __init__(
        self,
        num_oscillators: int,
        output_size: int,
        output_mapping_width: int,
        output_mapping_depth: int,
        *,
        key: Array,
    ) -> None:
        self.num_oscillators = num_oscillators
        self.output_shape = (
            2 * num_oscillators
        )  # The naming might be misleading as the output shape is actually the input shape
        self.output_mapping = mlp_init(
            in_size=self.output_shape,
            out_size=output_size,
            width_size=output_mapping_width,
            depth=output_mapping_depth,
            final_std=0.01,
            key=key,
        )

    def __call__(self, y: Array, x: Array, *, key: Array | None = None) -> Array:
        return self.output_mapping(cpg_output(y, self.num_oscillators))


class CPGNeuralActor(AbstractNeuralActor[ForcedCPG, CPGOutputMap], strict=True):
    """Neural actor with a CPG inductive bias"""

    num_oscillators: int
    convergence_factor: float
    vector_field: ForcedCPG
    output_mapping: CPGOutputMap
    initial_state_mapping: eqx.nn.MLP

    def __init__(
        self,
        num_oscillators: int,
        convergence_factor: float,
        input_size: int,
        input_mapping_width: int,
        input_mapping_depth: int,
        output_size: int,
        output_mapping_width: int,
        output_mapping_depth: int,
        *,
        key: Array,
    ) -> None:
        input_key, output_key, initial_state_key = jr.split(key, 3)

        self.num_oscillators = num_oscillators
        self.convergence_factor = convergence_factor

        self.vector_field = ForcedCPG(
            num_oscillators,
            convergence_factor,
            input_size,
            input_mapping_width,
            input_mapping_depth,
            key=input_key,
        )

        self.output_mapping = CPGOutputMap(
            num_oscillators,
            output_size,
            output_mapping_width,
            output_mapping_depth,
            key=output_key,
        )

        self.initial_state_mapping = mlp_init(
            in_size=input_size,
            out_size=self.vector_field.state_shape,
            width_size=input_mapping_width,
            depth=input_mapping_depth,
            key=initial_state_key,
        )

    @property
    def param_shape(self) -> tuple[int]:
        return (self.vector_field.param_shape,)

    @property
    def output_shape(self) -> tuple[int]:
        return (self.output_mapping.output_shape,)

    def initial_state(self, x: Array | None = None, *, key: Array | None) -> Array:
        assert x is not None, "Initial state requires an input"
        return self.initial_state_mapping(x)
