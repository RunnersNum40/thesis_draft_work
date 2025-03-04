"""Equinox implementation of a Neural Central Pattern Generator (CPG) model."""

import diffrax
import equinox as eqx
import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)


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
    t: float,
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


class ForcedCPG(eqx.Module):
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_oscillators = num_oscillators
        self.convergence_factor = convergence_factor
        self.state_shape = 3 * num_oscillators
        self.param_shape = 2 * (num_oscillators + num_oscillators**2)

        self.input_mapping = eqx.nn.MLP(
            in_size=input_size + 1 + self.state_shape,
            out_size=self.param_shape,
            width_size=input_mapping_width,
            depth=input_mapping_depth,
            activation=jax.nn.softplus,  # Continuously differentiable activation function theoretically required
            key=key,
        )

    def __call__(self, t: float, y: Array, x: Array) -> Array:
        params = self.input_mapping(jnp.concatenate([x, jnp.array([t]), y]))
        return cpg_vector_field(
            self.num_oscillators, self.convergence_factor, t, y, params
        )


class CPGOutputMap(eqx.Module):
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_oscillators = num_oscillators
        self.output_shape = (
            2 * num_oscillators
        )  # The naming might be misleading as the output shape is actually the input shape
        self.output_mapping = eqx.nn.MLP(
            in_size=self.output_shape,
            out_size=output_size,
            width_size=output_mapping_width,
            depth=output_mapping_depth,
            key=key,
        )

    def __call__(self, y: Array) -> Array:
        return self.output_mapping(cpg_output(y, self.num_oscillators))


class NeuralCPG(eqx.Module):
    """CPG dynamics incorporating external input and output mapping."""

    num_oscillators: int
    convergence_factor: float
    vector_field: ForcedCPG
    output_mapping: CPGOutputMap

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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        input_key, output_key = jax.random.split(key)

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

    def __call__(
        self, ts: Array, y0: Array, x: ArrayLike, map_output: bool = False
    ) -> tuple[Array, Array | None]:
        term = diffrax.ODETerm(self.vector_field)  # pyright: ignore
        solver = diffrax.Heun()
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = ts[1] - ts[0]
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-4)
        saveat = diffrax.SaveAt(t1=True)

        solution = diffrax.diffeqsolve(
            term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            args=x,
        )

        assert solution.ys is not None
        state = solution.ys[-1]

        if map_output:
            return state, self.output_mapping(state)

        return state, None
