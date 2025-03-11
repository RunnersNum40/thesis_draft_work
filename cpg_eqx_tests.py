"""Validation of the NeuralCPG model using Equinox."""

import logging

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

from cpg_eqx import CPGNeuralActor, cpg_output, cpg_vector_field

sns.set_theme(style="whitegrid")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

key = jr.key(1)

num_oscillators = 4
convergence_factor = 100.0
input_size = 2 * (num_oscillators + num_oscillators**2)
input_mapping_width = 16
input_mapping_depth = 2
output_size = 4
output_mapping_width = 16
output_mapping_depth = 0
key, model_key = jr.split(key)
model = CPGNeuralActor(
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

epochs = 2**13
lr = 1e-4
schedule = optax.schedules.cosine_decay_schedule(lr, epochs)
optimizer = optax.adabelief(learning_rate=schedule)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

data_length = 2**13


def data(key: Array) -> tuple[Array, Array, Array, Array]:
    params_key, state_key = jr.split(key)
    params = jr.uniform(
        params_key, (model.vector_field.param_shape,), minval=-1.0, maxval=1.0
    )
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
        max_steps=data_length * 2**4,
    )
    assert solution.ys is not None
    ys = solution.ys
    zs = jax.vmap(lambda y: cpg_output(y, num_oscillators)[:output_size])(ys)
    xs = jnp.ones(ts.shape + (input_size,))

    return ts, ys, xs, zs


key, data_key = jr.split(key)
ts, ys, xs, zs = data(data_key)
logger.debug(f"Data shapes: {ts.shape=}, {ys.shape=}, {xs.shape=}, {zs.shape=}")

key, split_key = jr.split(key)
train_indices = jnp.sort(
    jr.choice(key, data_length, (data_length // 4,), replace=False)
)
ts_train = ts[train_indices]
ys_train = ys[train_indices]
xs_train = xs[train_indices]
zs_train = zs[train_indices]
logger.debug(
    f"Train shapes: {ts_train.shape=}, {ys_train.shape=}, {xs_train.shape=}, {zs_train.shape=}"
)


def run_model_scan(
    model: CPGNeuralActor, ts: Array, y0: Array, xs: Array
) -> tuple[Array, Array]:
    def scan_fn(carry: Array, inp: tuple[Array, Array, Array]):
        t0, t1, x = inp

        new_carry, z_pred = model(ts=jnp.array([t0, t1]), y0=carry, x=x)
        return new_carry, (new_carry, z_pred)

    _, (ys_pred, zs_pred) = jax.lax.scan(scan_fn, y0, (ts[:-1], ts[1:], xs[:-1]))

    assert zs_pred is not None
    return ys_pred, zs_pred


@eqx.filter_value_and_grad
def grad_loss(
    model: CPGNeuralActor, ts: Array, y0: Array, xs: Array, zs: Array
) -> Array:
    _, zs_pred = run_model_scan(model, ts, y0, xs)
    return jnp.mean((zs[1:] - jnp.array(zs_pred)) ** 2)


def train(
    model: CPGNeuralActor,
    opt_state: optax.OptState,
    ts: Array,
    y0: Array,
    xs: Array,
    zs: Array,
    epochs: int,
    debug: bool = False,
) -> tuple[Array, CPGNeuralActor, optax.OptState]:
    arr, static = eqx.partition(model, eqx.is_array)

    @eqx.filter_jit
    def step(
        carry: tuple[Array, optax.OptState], i: Array
    ) -> tuple[tuple[Array, optax.OptState], Array]:
        arr, opt_state = carry
        model: CPGNeuralActor = eqx.combine(arr, static)  # pyright: ignore
        loss, grads = grad_loss(model, ts, y0, xs, zs)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        if debug:
            jax.debug.callback(
                lambda *args, **kwargs: logger.debug(
                    "Epoch: {epoch:04} Loss: {loss:e}".format(*args, **kwargs)
                ),
                epoch=i,
                loss=loss,
                ordered=True,
            )
        arr, _ = eqx.partition(model, eqx.is_array)
        return (arr, opt_state), loss

    (arr, opt_state), losses = jax.lax.scan(step, (arr, opt_state), jnp.arange(epochs))
    model = eqx.combine(arr, static)

    return losses, model, opt_state


losses, model, opt_state = train(
    model, opt_state, ts_train, ys_train[0], xs_train, zs_train, epochs, debug=True
)

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.yscale("log")
plt.title("Training Loss")
plt.show()


ys_pred, zs_pred = run_model_scan(model, ts, ys[0], xs)
loss, _ = grad_loss(model, ts, ys[0], xs, zs)
logger.info(f"Final Loss: {loss}")

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
