import numpy as np
from matplotlib import pyplot as plt


def square(p, half_size) -> float | np.ndarray:
    edge_distance = np.abs(p) - half_size
    return np.linalg.norm(np.maximum(edge_distance, 0)) + np.min(
        np.max(edge_distance), 0
    )


def ellipse(p, a, b) -> float | np.ndarray:
    return np.linalg.norm(p / np.array([a, b])) - 1.0


def plot_shape(func, bounds, resolution=100, *args, **kwargs):
    x = np.linspace(-bounds, bounds, resolution)
    y = np.linspace(-bounds, bounds, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda x, y: func(np.array([x, y]), *args, **kwargs))(X, Y)

    cs = plt.contourf(X, Y, Z, levels=resolution, cmap="coolwarm", alpha=0.75)
    plt.contour(X, Y, Z, levels=[0], colors="black", linewidths=1.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal")

    return cs


if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    cs = plot_shape(square, bounds=2, half_size=1.0)
    plt.title("Square")
    fig.colorbar(cs)

    plt.subplot(1, 2, 2)
    cs = plot_shape(ellipse, bounds=2, a=1, b=0.5)
    plt.title("Ellipse")
    fig.colorbar(cs)

    plt.tight_layout()
    plt.show()
