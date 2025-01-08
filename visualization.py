import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cpg import CPGState


def plot_trajectory(states: list[CPGState], dt: float):
    phases = np.array([state.phase for state in states])
    amplitudes = np.array([state.amplitude for state in states])

    plt.figure(figsize=(10, 5))
    for i in range(phases.shape[1]):
        plt.plot(np.arange(len(states)) * dt, phases[:, i], label=f"Phase {i+1}")
        plt.plot(
            np.arange(len(states)) * dt,
            amplitudes[:, i],
            "--",
            label=f"Amplitude {i+1}",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("CPG Oscillator Phase and Amplitude Over Time")
    plt.legend()
    plt.show()


def plot_polar_trajectory(states: list[CPGState]):
    phases = np.array([state.phase for state in states])
    amplitudes = np.array([state.amplitude for state in states])

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    for i in range(phases.shape[1]):
        ax.plot(phases[:, i] % (2 * np.pi), amplitudes[:, i], label=f"Oscillator {i+1}")

    ax.set_rmax(np.max(amplitudes) * 1.2)
    ax.set_title("Polar Representation of CPG Trajectory")
    plt.legend()
    plt.show()


def animate_trajectory(states: list[CPGState]):
    phases = np.array([state.phase for state in states])
    amplitudes = np.array([state.amplitude for state in states])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    lines = []

    for i in range(phases.shape[1]):
        (line,) = ax.plot([], [], label=f"Oscillator {i+1}")
        lines.append(line)

    ax.set_rmax(np.max(amplitudes) * 1.2)
    ax.set_title("Animated CPG Oscillator Trajectory")
    ax.legend()

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(phases[:frame, i] % (2 * np.pi), amplitudes[:frame, i])
        return lines

    ani = FuncAnimation(fig, update, frames=len(states), blit=True, interval=0.1)
    plt.show()
