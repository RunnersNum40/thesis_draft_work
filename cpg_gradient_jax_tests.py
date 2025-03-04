from typing import Generator

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from jax import Array
from jax import random as jr

from cpg_gradient_jax import CPGNetwork
from tqdm_rich_without_warnings import trange


class CPGPredictor(nnx.Module):
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
        self.cpg_network = CPGNetwork(
            num_oscillators=num_oscillators,
            convergence_factor=convergence_factor,
            input_layers=input_layers,
            output_layers=output_layers,
            solver=solver,
            rngs=rngs,
        )

    def __call__(self, ts: Array, y0: Array) -> Array:
        def step(state, t_pair):
            t0, t1 = t_pair
            timestep = t1 - t0
            return self.cpg_network(state, y0, t0, timestep, map_output=True)

        state = jnp.zeros(self.cpg_network.state_shape)
        _, y_preds = jax.lax.scan(step, state, jnp.stack([ts[:-1], ts[1:]], axis=1))
        return y_preds[-1]  # pyright: ignore


def _get_data(ts: Array, *, key: Array) -> Array:
    """Return time-series data for a damped harmonic oscillator."""
    solver = diffrax.Tsit5()
    y0 = jr.uniform(key, (2,), minval=-1.0, maxval=1.0)
    dt0 = 0.1

    zeta = 0.1
    omega0 = 1.0

    def damped_oscillator(t: float, state: Array, args) -> Array:
        """Vector field for a damped harmonic oscillator."""
        x, v = state
        dxdt = v
        dvdt = -2 * zeta * omega0 * v - omega0**2 * x
        return jnp.array([dxdt, dvdt])

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(damped_oscillator),  # pyright: ignore
        solver=solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
    )
    assert solution.ys is not None
    return solution.ys


def get_data(dataset_size: int, *, key: Array) -> tuple[Array, Array]:
    ts = jnp.linspace(0.0, 10.0, 100)
    keys = jr.split(key, dataset_size)
    ys = jax.vmap(_get_data, in_axes=None)(ts, key=keys)
    return ts, ys


def dataloader(
    array: Array, batch_size: int, *, key: Array
) -> Generator[Array, None, None]:
    dataset_size = array.shape[0]
    indices = jnp.arange(dataset_size)

    while True:
        key, batch_key = jr.split(key)
        batch_indices = jr.permutation(batch_key, indices)

        for i in range(0, dataset_size, batch_size):
            yield array[batch_indices[i : i + batch_size]]


def main(
    dataset_size: int = 128,
    batch_size: int = 32,
    lr: float = 1e-4,
    anneal_lr: bool = True,
    epochs: int = 1000,
    seed: int = 0,
    num_oscillators: int = 2,
    convergence_factor: float = 1e3,
    input_layers: list[int] = [2, 32],
    output_layers: list[int] = [32, 2],
) -> None:
    key = jr.key(seed)
    data_key, model_key, batches_key, key = jr.split(key, 4)
    rngs = nnx.Rngs(model_key)

    ts, ys = get_data(dataset_size, key=data_key)

    model = CPGPredictor(
        num_oscillators=num_oscillators,
        convergence_factor=convergence_factor,
        input_layers=input_layers,
        output_layers=output_layers,
        rngs=rngs,
        solver=diffrax.Tsit5,
    )
    if anneal_lr:
        schedule = optax.schedules.cosine_decay_schedule(lr, epochs)
    else:
        schedule = optax.constant_schedule(lr)
    optimizer = nnx.Optimizer(model, optax.adabelief(learning_rate=schedule))

    @nnx.value_and_grad
    def loss_grad(model: CPGPredictor, ti: Array, yi: Array) -> Array:
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi[:, -1] - y_pred) ** 2)

    @nnx.jit
    def step(
        model: CPGPredictor, optimizer: nnx.Optimizer, ti: Array, yi: Array
    ) -> Array:
        loss, grads = loss_grad(model, ti, yi)
        optimizer.update(grads)
        return loss

    losses = []
    for epoch, _ys in zip(trange(epochs), dataloader(ys, batch_size, key=batches_key)):
        loss = step(model, optimizer, ts, _ys)
        losses.append(loss)
        print(f"Epoch: {epoch:03} - Loss: {loss:.4f}")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


main(epochs=500, lr=1e-5)
