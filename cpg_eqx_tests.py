"""Validation of the NeuralCPG model using Equinox."""

import logging
from typing import Generator

import diffrax
import equinox as eqx
import jax
import numpy as np
import optax
import seaborn as sns
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from cpg_eqx import NeuralCPG, cpg_output, cpg_vector_field

sns.set_theme(style="whitegrid")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

key = jr.key(1)
key, data_key, model_key = jr.split(key, 3)

num_oscillators = 1
convergence_factor = 100.0
input_size = 2 * (num_oscillators + num_oscillators**2)
input_mapping_width = 16
input_mapping_depth = 4
output_size = 2
output_mapping_width = 0
output_mapping_depth = 0
model = NeuralCPG(
    num_oscillators=num_oscillators,
    convergence_factor=convergence_factor,
    input_size=input_size,
    input_mapping_width=input_mapping_width,
    input_mapping_depth=input_mapping_depth,
    output_size=output_size,
    output_mapping_width=output_mapping_width,
    output_mapping_depth=output_mapping_depth,
    key=model_key,
)

epochs = 4
lr = 1e-4
schedule = optax.schedules.cosine_decay_schedule(lr, epochs)
optimizer = optax.adabelief(learning_rate=schedule)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

data_length = 512
data_points = 32
batch_size = 4


@eqx.filter_jit
def data(key: Array) -> tuple[Array, Array, Array, Array]:
    params_key, state_key = jr.split(key)
    params = jr.normal(params_key, (model.vector_field.param_shape,))
    y0 = jr.normal(state_key, (model.vector_field.state_shape,))
    ts = jnp.linspace(0, 10, data_length)
    term = diffrax.ODETerm(
        lambda t, y, params: cpg_vector_field(
            num_oscillators,
            convergence_factor,
            t,  # pyright: ignore
            y,
            params,
        )
    )
    solution = diffrax.diffeqsolve(
        terms=term,
        solver=diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        y0=y0,
        dt0=ts[1] - ts[0],
        args=params,
        saveat=diffrax.SaveAt(ts=ts),
    )
    assert solution.ys is not None
    ys = solution.ys
    zs = jax.vmap(lambda y: cpg_output(y, num_oscillators)[:output_size])(ys)
    xs = params * jnp.ones((data_length, input_size))

    return ts, ys, xs, zs


ts, ys, xs, zs = jax.vmap(data)(jr.split(data_key, data_points))
logger.debug(f"Data shapes: {ts.shape=}, {ys.shape=}, {xs.shape=}, {zs.shape=}")


@eqx.filter_jit
def run_model_scan(
    model: NeuralCPG, ts: Array, y0: Array, xs: Array
) -> tuple[Array, Array]:
    def scan_fn(carry, inp):
        t0, t1, x = inp

        new_carry, z_pred = model(
            ts=jnp.array([t0, t1]), y0=carry, x=x, map_output=True
        )
        return new_carry, (new_carry, z_pred)  # Store carry history

    _, (ys_pred, zs_pred) = jax.lax.scan(scan_fn, y0, (ts[:-1], ts[1:], xs[:-1]))

    assert zs_pred is not None
    return ys_pred, zs_pred


@eqx.filter_value_and_grad
def grad_loss(model: NeuralCPG, ts: Array, y0: Array, xs: Array, zs: Array) -> Array:
    _, zs_pred = jax.vmap(lambda ts, y0, x: run_model_scan(model, ts, y0, x))(
        ts, y0, xs
    )
    return jnp.mean((zs[:, 1:] - jnp.array(zs_pred)) ** 2) / len(zs)


@eqx.filter_jit
def step(
    model: NeuralCPG,
    opt_state: optax.OptState,
    ts: Array,
    y0: Array,
    xs: Array,
    zs: Array,
) -> tuple[Array, NeuralCPG, optax.OptState]:
    loss, grads = grad_loss(model, ts, y0, xs, zs)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    return loss, new_model, opt_state


def batches(
    arrays: list[Array], batch_size: int, *, key: Array
) -> Generator[list[Array], None, None]:
    dataset_size = arrays[0].shape[0]
    assert all(
        array.shape[0] == dataset_size for array in arrays
    ), "Cannot batch arrays with different sizes"

    key, batch_key = jr.split(key)
    batch_indices = jr.permutation(batch_key, dataset_size, independent=True)

    for i in range(0, dataset_size, batch_size):
        yield [array[batch_indices[i : i + batch_size]] for array in arrays]


with logging_redirect_tqdm():
    for epoch in trange(epochs):
        key, batch_key = jr.split(key)
        losses = []
        for n, batch in enumerate(batches([ts, ys, xs, zs], batch_size, key=batch_key)):
            loss, model, opt_state = step(model, opt_state, ts, ys[:, 0], xs, zs)
            losses.append(loss)
            logger.debug(f"Batch {n:03}, Loss: {loss:0.6}")

        logger.info(f"\nEpoch {epoch:03}, Loss: {jnp.asarray(losses).mean():0.6}")


key, data_key = jr.split(key)
ts, ys, xs, zs = data(data_key)
ys_pred, zs_pred = run_model_scan(model, ts, ys[0], xs)

ts = np.array(ts[1:])
ys = np.array(ys[1:])
ys_pred = np.array(ys_pred)
zs = np.array(zs[1:])
zs_pred = np.array(zs_pred)

num_lines = zs.shape[1]
colors = plt.cm.jet(np.linspace(0, 1, num_lines))  # pyright: ignore

plt.figure(figsize=(10, 6))

for i in range(num_lines):
    plt.plot(ts, zs[:, i], color=colors[i], label=f"Ground Truth {i}")

for i in range(num_lines):
    plt.plot(ts, zs_pred[:, i], "--", color=colors[i], label=f"Predicted {i}")

plt.legend()
plt.title("CPG Model Outputs")
plt.show()

plt.figure(figsize=(10, 6))

for i in range(num_lines):
    plt.plot(ts, ys[:, i], color=colors[i], label=f"Ground Truth {i}")

for i in range(num_lines):
    plt.plot(ts, ys_pred[:, i], "--", color=colors[i], label=f"Predicted {i}")

plt.legend()
plt.title("CPG States")
plt.show()

plt.figure(figsize=(10, 6))

zs_pred = jax.vmap(lambda y: cpg_output(y, num_oscillators)[:output_size])(ys_pred)

for i in range(num_lines):
    plt.plot(ts, zs[:, i], color=colors[i], label=f"Ground Truth {i}")

for i in range(num_lines):
    plt.plot(ts, zs_pred[:, i], "--", color=colors[i], label=f"Predicted {i}")

plt.legend()
plt.title("CPG Outputs")
plt.show()
