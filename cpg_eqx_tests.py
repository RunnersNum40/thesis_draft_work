"""Validation of the NeuralCPG model using Equinox."""

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

from cpg_eqx import NeuralCPG, cpg_output, cpg_vector_field
from tqdm_rich_without_warnings import trange

sns.set_theme(style="whitegrid")

num_oscillators = 1
convergence_factor = 100.0

key = jr.key(1)
key, data_key, model_key, batch_key = jr.split(key, 4)

input_size = 1
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
epochs = 1000
lr = 5e-3
schedule = optax.schedules.cosine_decay_schedule(lr, epochs)
optimizer = optax.adabelief(learning_rate=schedule)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


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
    _, zs_pred = run_model_scan(model, ts, y0, xs)
    return jnp.mean((zs[1:] - jnp.array(zs_pred)) ** 2)


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


@eqx.filter_jit
def data(key: Array) -> tuple[Array, Array, Array, Array]:
    params_key, state_key = jr.split(key)
    params = jr.normal(params_key, (model.vector_field.param_shape,))
    y0 = jr.normal(state_key, (model.vector_field.state_shape,))
    ts = jnp.linspace(0, 10, 1000)
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
    xs = jnp.zeros(ts.shape + (input_size,))

    return ts, ys, xs, zs


def batches(
    arrays: list[Array], batch_size: int, *, key: Array
) -> Generator[list[Array], None, None]:
    dataset_size = arrays[0].shape[0]
    assert all(
        array.shape[0] == dataset_size for array in arrays
    ), "Cannot batch arrays with different sizes"
    indices = jnp.arange(dataset_size)

    while True:
        key, batch_key = jr.split(key)
        batch_indices = jr.permutation(batch_key, indices)

        for i in range(0, dataset_size, batch_size):
            yield [array[batch_indices[i : i + batch_size]] for array in arrays]


ts, ys, xs, zs = data(data_key)
for epoch in trange(epochs):
    loss, model, opt_state = step(model, opt_state, ts, ys[0], xs, zs)
    print(f"Epoch {epoch:03}, Loss: {loss:0.6}")


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
plt.title("Ground Truth vs Predicted Output")
plt.xlabel("Time")
plt.ylabel("Output")
plt.show()

num_lines = ys.shape[1]
colors = plt.cm.jet(np.linspace(0, 1, num_lines))  # pyright: ignore

plt.figure(figsize=(10, 6))

for i in range(num_lines):
    plt.plot(ts, ys[:, i], color=colors[i], label=f"Ground Truth {i}")

for i in range(num_lines):
    plt.plot(ts, ys_pred[:, i], "--", color=colors[i], label=f"Predicted {i}")

plt.legend()
plt.title("Ground Truth vs Predicted State")
plt.xlabel("Time")
plt.ylabel("State")
plt.show()
