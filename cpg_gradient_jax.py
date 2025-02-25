from typing import Callable

import diffrax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import ArrayLike


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


class MLP(nnx.Module):
    def __init__(
        self,
        layers: list[int],
        activation_function: Callable[[ArrayLike], Array],
        final_layer_kernel_scale: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.activation_function = activation_function
        self.layers = tuple(
            nnx.Linear(
                din,
                dout,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(
                    jnp.sqrt(2.0) if n != len(layers) - 1 else final_layer_kernel_scale
                ),
                bias_init=nnx.initializers.constant(jnp.array(0.0)),
            )
            for n, (din, dout) in enumerate(zip(layers[:-1], layers[1:]))
        )

    def __call__(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x)

        for layer in self.layers:
            x = layer(x)
            x = self.activation_function(x)
        return x


class CPG(nnx.Module):
    """Central Pattern Generator (CPG) model.

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

    def __init__(self, num_oscillators: int, convergence_factor: float) -> None:
        self.num_oscillators = num_oscillators
        self.convergence_factor = convergence_factor
        self.state_shape = (3 * num_oscillators,)
        self.param_shape = (2 * (num_oscillators + num_oscillators**2),)

    def __call__(self, t: float, state: ArrayLike, args: ArrayLike) -> Array:
        amplitudes, phases, amplitudes_dot = split_states(state, self.num_oscillators)
        intrinsic_amplitudes, intrinsic_frequencies, coupling_weights, phase_biases = (
            split_params(args, self.num_oscillators)
        )

        phase_diffs = jnp.subtract.outer(phases, phases) - phase_biases
        coupling_terms = (
            amplitudes.reshape(-1, 1) * coupling_weights * jnp.sin(phase_diffs)
        )
        phase_dot = intrinsic_frequencies + coupling_terms.sum(axis=1)

        amplitudes_dot_dot = self.convergence_factor * (
            (self.convergence_factor / 4) * (intrinsic_amplitudes - amplitudes)
            - amplitudes_dot
        )

        return jnp.concatenate([amplitudes_dot, phase_dot, amplitudes_dot_dot])


class CPGNetwork(nnx.Module):
    final_input_layer_kernel_scale: float = 1.0
    final_output_layer_kernel_scale: float = 0.01

    def __init__(
        self,
        num_oscillators: int,
        convergence_factor: float,
        input_layers: list[int],
        output_layers: list[int],
        solver: type[diffrax.AbstractSolver] = diffrax.Dopri5,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_oscillators = num_oscillators
        self.solver = solver

        self.cpg = CPG(num_oscillators, convergence_factor)

        input_layers[0] += 1  # time input
        input_layers[0] += 3 * num_oscillators  # state input
        input_layers.append(self.cpg.param_shape[0])  # params
        self.input_network = MLP(
            input_layers, nnx.relu, self.final_input_layer_kernel_scale, rngs=rngs
        )

        output_layers.insert(0, num_oscillators * 2)  # state
        self.output_network = MLP(
            output_layers, nnx.relu, self.final_output_layer_kernel_scale, rngs=rngs
        )

    def __call__(
        self, state: ArrayLike, x: ArrayLike, time: float, timestep: float
    ) -> tuple[Array, Array]:
        term = diffrax.ODETerm(self.cpg)  # pyright: ignore
        solver = self.solver()
        stepsize_controller = (
            diffrax.PIDController(rtol=1e-5, atol=1e-5)
            if isinstance(solver, diffrax.AbstractAdaptiveSolver)
            else diffrax.ConstantStepSize()
        )
        ts = [time, time + timestep]
        saveat = diffrax.SaveAt(ts=ts)

        x = jnp.concatenate([jnp.asarray([time]), state, x])
        params = nnx.tanh(self.input_network(x))

        solution = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=timestep,
            y0=state,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            args=params,
        )

        assert solution.ys is not None

        state = jnp.asarray(solution.ys[0])

        amplitudes, phases, _ = split_states(state, self.num_oscillators)
        cpg_output = jnp.concatenate(
            [amplitudes * jnp.sin(phases), amplitudes * jnp.cos(phases)]
        )

        return state, self.output_network(cpg_output)

    @property
    def state_shape(self) -> tuple[int]:
        return self.cpg.state_shape

    @property
    def param_size(self) -> tuple[int]:
        return self.cpg.param_shape
