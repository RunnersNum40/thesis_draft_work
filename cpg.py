"""
Central Pattern Generator (CPG) model.

Based on the model described in:
Ijspeert, A.J. & Crespi, Alessandro & Ryczko, Dimitri & Cabelguen,
jean-marie. (2007). From Swimming to Walking with a Salamander
Robot Driven by a Spinal Cord Model. Science (New York, N.Y.). 315.
1416-20. 10.1126/science.1138353.
"""

from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp


class CPGParams(NamedTuple):
    intrinsic_amplitude: np.ndarray
    intrinsic_frequency: np.ndarray
    convergence_factor: np.ndarray
    coupling_strength: np.ndarray
    phase_bias: np.ndarray


class CPGState(NamedTuple):
    phase: np.ndarray
    amplitude: np.ndarray
    amplitude_dot: np.ndarray


def cpg_ode(t, y, params: CPGParams):
    n = len(params.intrinsic_amplitude)
    phase = y[:n]
    amplitude = y[n : 2 * n]
    amplitude_dot = y[2 * n :]

    amplitude_dot_dot = params.convergence_factor * (
        params.convergence_factor / 4 * (params.intrinsic_amplitude - amplitude)
        - amplitude_dot
    )

    phase_dot = params.intrinsic_frequency.copy()
    for i in range(n):
        for j in range(n):
            phase_dot[i] += (
                amplitude[j]
                * params.coupling_strength[i, j]
                * np.sin(phase[j] - phase[i] + params.phase_bias[i, j])
            )

    dydt = np.concatenate([phase_dot, amplitude_dot, amplitude_dot_dot])
    return dydt


def step_cpg(state: CPGState, params: CPGParams, dt: float) -> CPGState:
    n = len(state.phase)
    y0 = np.concatenate([state.phase, state.amplitude, state.amplitude_dot])
    t_span = [0, dt]

    sol = solve_ivp(cpg_ode, t_span, y0, args=(params,))

    phase = sol.y[:n, -1]
    amplitude = sol.y[n : 2 * n, -1]
    amplitude_dot = sol.y[2 * n :, -1]

    return CPGState(phase=phase, amplitude=amplitude, amplitude_dot=amplitude_dot)


if __name__ == "__main__":
    from visualization import animate_trajectory, plot_polar_trajectory, plot_trajectory
    from tqdm.rich import tqdm

    intrinsic_amplitude = np.array([2.0, 1.0])
    intrinsic_frequency = np.array([2.0 * np.pi, -2.0 * np.pi])
    convergence_factor = np.array([10.0, 10.0])
    coupling_strength = np.array([[0.0, 1.0], [1.0, 0.0]])
    phase_bias = np.array([[0.0, np.pi / 2], [-np.pi / 2, 0.0]])

    params = CPGParams(
        intrinsic_amplitude=intrinsic_amplitude,
        intrinsic_frequency=intrinsic_frequency,
        convergence_factor=convergence_factor,
        coupling_strength=coupling_strength,
        phase_bias=phase_bias,
    )

    initial_phase = np.array([0.0, 0.0])
    initial_amplitude = np.array([1.0, 2.0])
    initial_amplitude_dot = np.array([0.0, 0.0])

    state = CPGState(
        phase=initial_phase,
        amplitude=initial_amplitude,
        amplitude_dot=initial_amplitude_dot,
    )

    dt = 1e-3
    steps = 1000
    states = []

    for i in tqdm(range(steps)):
        state = step_cpg(state, params, dt)
        states.append(state)

    states_and_params = list(zip(states, [params] * len(states)))
    plot_trajectory(states_and_params, dt)
    plot_polar_trajectory(states_and_params)
    animate_trajectory(states_and_params)
