import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cpg import CPGParams, CPGState


def plot_trajectory(
    states_and_params: list[tuple[CPGState, CPGParams]],
    dt: float,
    show_params: bool = True,
):
    phases = np.mod(
        np.array([state.phase for state, params in states_and_params]), np.pi
    )
    amplitudes = np.array([state.amplitude for state, params in states_and_params])
    intrinsic_amplitudes = np.array(
        [params.intrinsic_amplitude for state, params in states_and_params]
    )
    intrinsic_frequencies = np.array(
        [params.intrinsic_frequency for state, params in states_and_params]
    )

    plt.figure(figsize=(10, 5))
    for i in range(phases.shape[1]):
        plt.plot(
            np.arange(len(states_and_params)) * dt, phases[:, i], label=f"Phase {i+1}"
        )
        plt.plot(
            np.arange(len(states_and_params)) * dt,
            amplitudes[:, i],
            label=f"Amplitude {i+1}",
        )
        if show_params:
            plt.plot(
                np.arange(len(states_and_params)) * dt,
                intrinsic_amplitudes[:, i],
                "--",
                label=f"Intrinsic Amplitude {i+1}",
                alpha=0.5,
            )
            plt.plot(
                np.arange(len(states_and_params)) * dt,
                intrinsic_frequencies[:, i],
                "--",
                label=f"Intrinsic Frequency {i+1}",
                alpha=0.5,
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_polar_trajectory(
    states_and_params: list[tuple[CPGState, CPGParams]],
    show_params: bool = True,
):
    phases = np.array([state.phase for state, params in states_and_params])
    amplitudes = np.array([state.amplitude for state, params in states_and_params])
    intrinsic_amplitudes = np.array(
        [params.intrinsic_amplitude for state, params in states_and_params]
    )

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    for i in range(phases.shape[1]):
        ax.plot(phases[:, i] % (2 * np.pi), amplitudes[:, i], label=f"Oscillator {i+1}")
        if show_params:
            ax.plot(
                phases[:, i] % (2 * np.pi),
                intrinsic_amplitudes[:, i],
                "--",
                label=f"Intrinsic Amplitude {i+1}",
                alpha=0.5,
            )

    ax.set_rmin(0)
    ax.set_rmax(np.max(amplitudes) * 1.2)
    plt.legend()
    plt.show()


def animate_trajectory(states_and_params: list[tuple[CPGState, CPGParams]]):
    phases = np.array([state.phase for state, params in states_and_params])
    amplitudes = np.array([state.amplitude for state, params in states_and_params])
    intrinsic_amplitudes = np.array(
        [params.intrinsic_amplitude for state, params in states_and_params]
    )

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    lines = []

    for i in range(phases.shape[1]):
        (line,) = ax.plot([], [], label=f"Oscillator {i+1}")
        lines.append(line)

    ax.set_rmin(0)
    ax.set_rmax(np.max(amplitudes) * 1.2)
    ax.legend()

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(phases[:frame, i] % (2 * np.pi), amplitudes[:frame, i])
        return lines

    ani = FuncAnimation(
        fig, update, frames=len(states_and_params), blit=True, interval=100 / len(lines)
    )
    plt.show()
