import matplotlib.pyplot as plt
import numpy as np
from tqdm.rich import tqdm

from cpg import CPGParams, CPGState, step_cpg

intrinsic_amplitude = np.array([2.0])
intrinsic_frequency = np.array([2.0 * np.pi])
convergence_factor = np.array([10.0])
coupling_strength = np.array([[0.0]])
phase_bias = np.array([[0.0]])

params = CPGParams(
    intrinsic_amplitude=intrinsic_amplitude,
    intrinsic_frequency=intrinsic_frequency,
    convergence_factor=convergence_factor,
    coupling_strength=coupling_strength,
    phase_bias=phase_bias,
)

initial_phase = np.array([0.0])
initial_amplitude = np.array([1.0])
initial_amplitude_dot = np.array([0.0])

state = CPGState(
    phase=initial_phase,
    amplitude=initial_amplitude,
    amplitude_dot=initial_amplitude_dot,
)

dt = 1e-2
steps = 1000
states = []

for i in tqdm(range(steps)):
    state = step_cpg(state, params, dt)
    states.append(state)

h = 1
d = 1
g_c = 1
g_p = 0.1

x = np.array([-d * (state.amplitude - 1) * np.cos(state.phase) for state in states])
y = np.array([-d * (state.amplitude - 1) * np.sin(state.phase) for state in states])
z = np.array([np.sin(state.phase) for state in states])
z[z > 0] *= g_c
z[z < 0] *= g_p

plt.plot(x[:, 0], z[:, 0], label="Oscillator 1")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
