import logging
from typing import Literal

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

logger = logging.getLogger(__name__)

# 64-bit precision required for numerical stability
logging.info("Setting JAX to 64-bit precision")
jax.config.update("jax_enable_x64", True)


class TensorMLP(eqx.Module):
    """Modification of the MLP class from Equinox to handle tensors as input and output."""

    in_shape: tuple[int, ...] | Literal["scalar"]
    out_shape: tuple[int, ...] | Literal["scalar"]
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_shape: tuple[int, ...] | Literal["scalar"],
        width_size: int,
        depth: int,
        out_shape: tuple[int, ...] | Literal["scalar"],
        *,
        key: Array,
        **kwargs,
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape

        if in_shape == "scalar":
            in_size = "scalar"
        else:
            in_size = int(jnp.asarray(in_shape).prod())

        if out_shape == "scalar":
            out_size = "scalar"
        else:
            out_size = int(jnp.asarray(out_shape).prod())

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
            **kwargs,
        )

    def __call__(self, x: Array) -> Array:
        if self.in_shape == "scalar":
            x = self.mlp(x)
        else:
            x = jnp.ravel(x)
            x = self.mlp(x)

        if self.out_shape != "scalar":
            x = jnp.reshape(x, self.out_shape)

        return x


class Field(eqx.Module):
    """Version of a TensorMLP that takes extra parameters to support use as a field."""

    tensor_mlp: TensorMLP

    def __init__(
        self,
        in_shape: tuple[int, ...] | Literal["scalar"],
        width_size: int,
        depth: int,
        out_shape: tuple[int, ...] | Literal["scalar"],
        *,
        key: Array,
        **kwargs,
    ) -> None:
        self.tensor_mlp = TensorMLP(
            in_shape=in_shape,
            width_size=width_size,
            depth=depth,
            out_shape=out_shape,
            key=key,
            **kwargs,
        )

        # # Make the weights smaller to prevent blowing up
        # self.tensor_mlp = eqx.tree_at(
        #     lambda tree: [linear.weight for linear in tree.mlp.layers],
        #     self.tensor_mlp,
        #     replace_fn=lambda x: x / 10.0,
        # )

    def __call__(self, t: Array, x: Array, args) -> Array:
        return self.tensor_mlp(x)


class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    field: Field
    output: eqx.nn.MLP

    input_size: int | Literal["scalar"]
    state_size: int
    output_size: int | Literal["scalar"]

    def __init__(
        self,
        input_size: int | Literal["scalar"],
        hidden_size: int,
        output_size: int | Literal["scalar"],
        width_size: int,
        depth: int,
        *,
        key: Array,
        initial_width_size: int | None = None,
        initial_depth: int | None = None,
        output_width_size: int | None = None,
        output_depth: int | None = None,
    ) -> None:
        """
        Neural Controlled Differential Equation model.

        - input_size: The size of the input vector.
        - hidden_size: The dimension of the hidden state.
            Must be greater than or equal to the input size for the CDE to be a
            universal approximator.
        - output_size: The size of the output vector.
        - width_size: The width of the hidden layers.
            Not recommended to be less than the hidden size.
        - depth: The number of hidden layers.

        - key: The random key for initialization.
        - initial_width_size: The width of the hidden layers for the initial condition.
            If None, defaults to width_size.
        - initial_depth: The number of hidden layers for the initial condition.
            If None, defaults to 0 meaning a linear initial condition.
        - output_width_size: The width of the hidden layers for the output.
            If None, defaults to width_size.
        - output_depth: The number of hidden layers for the output.
            If None, defaults to 0 meaning a linear output.
        """
        ikey, fkey, okey = jr.split(key, 3)

        self.input_size = input_size
        self.state_size = hidden_size
        self.output_size = output_size

        if input_size == "scalar":
            input_size = 1

        # Defualt to a linear initial condition
        self.initial = eqx.nn.MLP(
            in_size=input_size + 1,
            out_size=hidden_size,
            width_size=initial_width_size or width_size,
            depth=initial_depth or 0,
            key=ikey,
        )

        # Matrix output of hidden_size x (input_size + 1)
        # Tanh to prevent blowing up
        self.field = Field(
            in_shape=(hidden_size,),
            out_shape=(hidden_size, input_size + 1),
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=jnn.tanh,
            key=fkey,
        )

        # Default to a linear output
        self.output = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=output_size,
            width_size=output_width_size or width_size,
            depth=output_depth or 0,
            key=okey,
        )

    def __call__(
        self,
        ts: Array,
        xs: Array,
        z0: Array,
        *,
        evolving_out: bool = False,
    ) -> tuple[Array, Array, diffrax.RESULTS]:
        """Compute the output of the Neural CDE.

        Evaluates the Neural CDE at a series of times and inputs using cubic
        hermite splines with backward differences.

        Arguments:
        - ts: The times to evaluate the Neural CDE.
        - xs: The inputs to the Neural CDE.
        - z0: The initial state of the Neural CDE.
        - evolving_out: If True, return the output at each time step.

        Returns:
        - z1: The final state of the Neural CDE.
        - y1: The output of the Neural CDE.
        """
        z0 = eqx.error_if(
            z0,
            jnp.any(jnp.isnan(z0)),
            "Initial state contains NaN",
        )
        ts = eqx.error_if(ts, jnp.all(jnp.isnan(ts)), "Times are all NaN")
        xs = eqx.error_if(xs, jnp.all(jnp.isnan(xs)), "Inputs are all NaN")

        if self.input_size == "scalar":
            assert xs.ndim == 1
            xs = jnp.expand_dims(xs, axis=-1)

        assert ts.ndim == 1
        assert xs.ndim == 2
        assert z0.ndim == 1
        assert xs.shape[0] == ts.shape[0]

        # If the input only has one timestep add an extra small timestep
        t1 = jnp.nanmax(ts)
        xs = jax.lax.cond(
            jnp.any(jnp.isnan(xs[1])),
            lambda: xs.at[1].set(xs[0]),
            lambda: xs,
        )
        ts = jax.lax.cond(
            jnp.any(jnp.isnan(ts[1])),
            lambda: ts.at[1].set(ts[0] + 1e-3),
            lambda: ts,
        )

        # Create a control term with a cubic interpolation of the input
        xs = jnp.concatenate([ts[:, None], xs], axis=1)  # Add time to input
        coeffs = diffrax.backward_hermite_coefficients(ts, xs)
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.field, control).to_ode()  # pyright: ignore

        initial_field = self.field(ts[0], z0, None)
        initial_control = control.evaluate(ts[0])
        initial_derivative = jnp.matmul(initial_field, initial_control)

        initial_field = eqx.error_if(
            initial_field,
            jnp.any(jnp.isnan(initial_field)),
            "Initial field contains NaN",
        )
        initial_control = eqx.error_if(
            initial_control,
            jnp.any(jnp.isnan(initial_control)),
            "Initial control contains NaN",
        )
        initial_derivative = eqx.error_if(
            initial_derivative,
            jnp.any(jnp.isnan(initial_derivative)),
            "Initial derivative contains NaN",
        )

        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

        if evolving_out:
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)

        solution = diffrax.diffeqsolve(
            term,
            solver=solver,
            t0=ts[0],
            t1=t1,
            dt0=None,
            y0=z0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )

        result = solution.result
        if evolving_out:
            zs = solution.ys
            zs = jnp.asarray(jnp.where(jnp.isnan(ts[:, None]), jnp.nan, zs))
            return zs, jax.vmap(self.output)(zs), result
        else:
            z1 = solution.ys[-1]
            return z1, self.output(z1), result

    def initial_state(self, t0: float | Array, x0: Array) -> Array:
        """Return an initial state for the given input and time.

        Arguments:
        - x0: The initial input.
        - t0: The initial time.

        Returns:
        - z0: The initial state.
        """
        t0 = jnp.asarray(t0)

        if t0.ndim < 1:
            t0 = jnp.expand_dims(t0, axis=-1)

        if self.input_size == "scalar":
            x0 = jnp.expand_dims(x0, axis=-1)

        assert t0.shape == (1,)
        assert x0.ndim == 1

        return self.initial(jnp.concatenate([t0, x0]))


if __name__ == "__main__":
    from typing import Any

    import jax.scipy as jsp
    import numpy as np
    import optax
    from matplotlib import pyplot as plt

    def random_matrix(n: int, min_eig: float, max_eig: float, *, key: Array) -> Array:
        """Return a random symmetric matrix with eigenvalues within a range."""
        key, eig_key, matrix_key = jr.split(key, 3)
        eigvals = jr.uniform(eig_key, (n,), minval=min_eig, maxval=max_eig)
        random_matrix = jr.normal(matrix_key, (n, n))
        Q, R = jnp.linalg.qr(random_matrix)
        diag = jnp.sign(jnp.diag(R))
        Q = Q * diag
        A = Q @ jnp.diag(eigvals) @ Q.T
        return A

    def get_data(
        n: int,
        length: int,
        t0: float,
        t1: float,
        *,
        add_noise: bool = True,
        noise_scale: float = 1e-3,
        key: Array,
    ) -> tuple[Array, Array, Array]:
        """Generate a linear trajectory.

        The dataset of a series of symmetric matrix exponetial products.
        The goal is to predict the diagonal of the matrix that generates the trajectory.
        """
        initial_key, matrix_key, noise_key = jr.split(key, 3)
        ts = jnp.linspace(t0, t1, length)

        x0 = jr.uniform(initial_key, (n,))
        matrix = random_matrix(n, -1, 1e-1, key=matrix_key)

        xs = jax.vmap(lambda t: jsp.linalg.expm(t * matrix) @ x0)(ts)
        if add_noise:
            xs = xs + jr.normal(noise_key, xs.shape) * noise_scale

        y1 = jnp.diag(matrix)

        return xs, ts, y1

    def batch_indices(dataset_size: int, batch_size: int, *, key: Array) -> Array:
        return jnp.reshape(jr.permutation(key, dataset_size), (-1, batch_size))

    dataset_size = 1024
    data_length = 128
    data_time = 4 * jnp.pi
    add_noise = False
    data_dim = 2
    epochs = 256
    batch_size = 32
    learning_rate = 1e-4
    seed = 0

    key = jr.key(seed)
    key, data_key, model_key = jr.split(key, 3)

    xs, ts, y1 = jax.vmap(
        lambda key, t0: get_data(
            data_dim,
            data_length,
            t0=t0,
            t1=t0 + data_time,
            add_noise=add_noise,
            key=key,
        )
    )(jr.split(data_key, dataset_size), jnp.linspace(0, 1, dataset_size))

    neural_cde = NeuralCDE(
        input_size=data_dim,
        hidden_size=8,
        output_size=data_dim,
        width_size=128,
        depth=2,
        key=model_key,
        initial_depth=2,
        output_depth=2,
    )

    schedule = optax.cosine_decay_schedule(
        learning_rate, epochs * dataset_size // batch_size
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(neural_cde, eqx.is_array))

    @eqx.filter_jit
    def loss(
        model: NeuralCDE, ts_batch: Array, xs_batch: Array, y1_batch: Array
    ) -> tuple[Array, Array]:
        """Compute the loss of a batch of control inputs and labels"""
        z0_batch = jax.vmap(model.initial_state)(ts_batch[:, 0], xs_batch[:, 0])
        _, prediction_batch, _ = jax.vmap(model)(ts_batch, xs_batch, z0_batch)
        error_batch = jax.vmap(jnp.linalg.norm)(prediction_batch - y1_batch)
        return jnp.mean(error_batch), prediction_batch

    grad_loss = eqx.filter_value_and_grad(loss, has_aux=True)

    @eqx.filter_jit
    def step(
        model: NeuralCDE,
        ts_batch: Array,
        xs_batch: Array,
        y1_batch: Array,
        opt_state: optax.OptState,
    ) -> tuple[NeuralCDE, optax.OptState, Array]:
        (loss, _), grads = grad_loss(model, ts_batch, xs_batch, y1_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    model_dynamic, model_static = eqx.partition(neural_cde, eqx.is_array)

    @eqx.filter_jit
    def scan_epoch(
        carry: tuple[Any, optax.OptState, Array, tuple[Array, Array, Array]], i
    ) -> tuple[tuple[Any, optax.OptState, Array, tuple[Array, Array, Array]], Array]:
        model_dynamic, opt_state, key, data = carry

        @eqx.filter_jit
        def scan_batch(
            carry: tuple[Any, optax.OptState], batch: Array
        ) -> tuple[tuple[Any, optax.OptState], Array]:
            model_dynamic, opt_state = carry
            ts_batch = ts[batch]
            xs_batch = xs[batch]
            y1_batch = y1[batch]

            model = eqx.combine(model_dynamic, model_static)
            model, opt_state, loss = step(
                model, ts_batch, xs_batch, y1_batch, opt_state
            )
            model_dynamic, _ = eqx.partition(model, eqx.is_array)

            return (model_dynamic, opt_state), loss

        key, batch_key = jr.split(key)
        (model_dynamic, opt_state), losses = jax.lax.scan(
            scan_batch,
            (model_dynamic, opt_state),
            batch_indices(dataset_size, batch_size, key=batch_key),
        )

        jax.debug.print(
            "Epoch: {epoch}, loss: {loss}", epoch=i, loss=losses.mean(), ordered=True
        )

        return (model_dynamic, opt_state, key, data), jnp.mean(losses)

    key, data_key = jr.split(key)
    (model_dynamic, _, _, _), losses = jax.lax.scan(
        scan_epoch,
        (model_dynamic, opt_state, data_key, (ts, xs, y1)),
        jnp.arange(epochs),
    )
    neural_cde: NeuralCDE = eqx.combine(model_dynamic, model_static)

    print(f"Final Train Loss: {losses[-1]}")

    key, test_key = jr.split(key)
    xs_test, ts_test, y1_test = jax.vmap(
        lambda key, t0: get_data(
            data_dim,
            data_length,
            t0=t0,
            t1=t0 + data_time,
            add_noise=add_noise,
            key=key,
        )
    )(jr.split(data_key, dataset_size), jnp.linspace(0, 1, dataset_size))
    _loss, prediction = loss(neural_cde, ts_test, xs_test, y1_test)

    print(f"Test Loss: {_loss}")
    print(f"Predicted Matrix: {prediction[0]}")
    print(f"True Matrix: {y1_test[0]}")

    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    _, prediction, _ = neural_cde(
        ts_test[0],
        xs_test[0],
        neural_cde.initial_state(ts_test[0][0], xs_test[0][0]),
        evolving_out=True,
    )

    ts = np.array(ts_test[0])
    xs = np.array(xs_test[0])
    prediction = np.array(prediction)
    y1 = np.array(y1_test[0])

    from matplotlib.collections import LineCollection

    def colored_trajectory(xy: np.ndarray, cmap: str, label: str) -> LineCollection:
        points = xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,  # pyright: ignore
            cmap=cmap,
            array=np.arange(len(segments)),
            label=label,
            linewidth=2,
        )
        return lc

    fig, ax = plt.subplots()

    ax.add_collection(colored_trajectory(xs, cmap="Blues", label="True"))
    ax.add_collection(colored_trajectory(prediction, cmap="Reds", label="Prediction"))
    ax.plot(y1[0], y1[1], "o", label="Target", color="green")

    ax.autoscale()
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    plt.show()
