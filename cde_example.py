import math

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
import optax


class Func(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            # Note the use of a softplus activation function. This is important to
            # ensure the vector field is continuously differentiable.
            activation=jnn.softplus,
            # Note the use of a tanh final activation function. This is important to
            # stop the model blowing up.
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lkey)

    def __call__(self, ts, coeffs, evolving_out=False):
        # Each sample of data consists of some timestamps `ts`, and some `coeffs`
        # parameterising a control path. These are used to produce a continuous-time
        # input path `control`.
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        solver = diffrax.Tsit5()
        dt0 = None
        y0 = self.initial(control.evaluate(ts[0]))
        if evolving_out:
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=saveat,
        )
        if evolving_out:
            prediction = jax.vmap(lambda y: jnn.sigmoid(self.linear(y))[0])(solution.ys)
        else:
            (prediction,) = jnn.sigmoid(self.linear(solution.ys[-1]))
        return prediction


def get_data(dataset_size, add_noise, *, key):
    theta_key, noise_key = jr.split(key, 2)
    length = 100
    theta = jr.uniform(theta_key, (dataset_size,), minval=0, maxval=2 * math.pi)
    y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    ts = jnp.broadcast_to(jnp.linspace(0, 4 * math.pi, length), (dataset_size, length))
    matrix = jnp.array([[-0.3, 2], [-2, -0.3]])
    ys = jax.vmap(
        lambda y0i, ti: jax.vmap(lambda tij: jsp.linalg.expm(tij * matrix) @ y0i)(ti)
    )(y0, ts)
    ys = jnp.concatenate([ts[:, :, None], ys], axis=-1)  # time is a channel
    ys = ys.at[: dataset_size // 2, :, 1].multiply(-1)
    if add_noise:
        ys = ys + jr.normal(noise_key, ys.shape) * 0.1
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
    labels = jnp.zeros((dataset_size,))
    labels = labels.at[: dataset_size // 2].set(1.0)
    _, _, data_size = ys.shape
    return ts, coeffs, labels, data_size


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size=256,
    add_noise=False,
    batch_size=32,
    lr=1e-3,
    steps=100,
    hidden_size=8,
    width_size=128,
    depth=1,
    seed=0,
):
    key = jr.key(seed)
    train_data_key, test_data_key, model_key, loader_key = jr.split(key, 4)

    ts, coeffs, labels, data_size = get_data(
        dataset_size, add_noise, key=train_data_key
    )

    model = NeuralCDE(data_size, hidden_size, width_size, depth, key=model_key)

    @eqx.filter_jit
    def loss(model, ti, label_i, coeff_i):
        pred = jax.vmap(model)(ti, coeff_i)
        # Binary cross-entropy
        bxe = label_i * jnp.log(pred) + (1 - label_i) * jnp.log(1 - pred)
        bxe = -jnp.mean(bxe)
        acc = jnp.mean((pred > 0.5) == (label_i == 1))
        return bxe, acc

    grad_loss = eqx.filter_value_and_grad(loss, has_aux=True)

    @eqx.filter_jit
    def make_step(model, data_i, opt_state):
        ti, label_i, *coeff_i = data_i
        (bxe, acc), grads = grad_loss(model, ti, label_i, coeff_i)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return bxe, acc, model, opt_state

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    for step, data_i in zip(
        range(steps), dataloader((ts, labels) + coeffs, batch_size, key=loader_key)
    ):
        bxe, acc, model, opt_state = make_step(model, data_i, opt_state)
        print(f"Step: {step}, Loss: {bxe}, Accuracy: {acc}")

    ts, coeffs, labels, _ = get_data(dataset_size, add_noise, key=test_data_key)
    bxe, acc = loss(model, ts, labels, coeffs)
    print(f"Test loss: {bxe}, Test Accuracy: {acc}")

    # Plot results
    sample_ts = ts[-1]
    sample_coeffs = tuple(c[-1] for c in coeffs)
    pred = model(sample_ts, sample_coeffs, evolving_out=True)
    interp = diffrax.CubicInterpolation(sample_ts, sample_coeffs)
    values = jax.vmap(interp.evaluate)(sample_ts)

    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sample_ts, values[:, 1], label="Data 1")
    ax1.plot(sample_ts, values[:, 2], label="Data 2")
    ax1.plot(sample_ts, pred, c="crimson", label="Classification")
    ax1.set_xlabel("t")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(values[:, 1], values[:, 2], label="Data")
    ax2.plot(values[:, 1], values[:, 2], pred, c="crimson", label="Classification")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("Classification")

    plt.tight_layout()
    plt.show()


main()
