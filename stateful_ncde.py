import logging
from abc import abstractmethod
from typing import Callable, ClassVar, Literal

import diffrax
import equinox as eqx
import jax
from jax import lax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Float, Int, Key, PyTree, ScalarLike

logger = logging.getLogger(__name__)

logger.info("Setting JAX to 64-bit precision")
jax.config.update("jax_enable_x64", True)


class TensorMLP(eqx.Module):
    """Modification of the MLP class from Equinox to handle tensors as input and output."""

    in_shape: tuple[int, ...] | Literal["scalar"]
    out_shape: tuple[int, ...] | Literal["scalar"]
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_shape: tuple[int, ...] | Literal["scalar"],
        out_shape: tuple[int, ...] | Literal["scalar"],
        width_size: int,
        depth: int,
        activation: Callable[[Array], Array] = jnn.relu,
        final_activation: Callable[[Array], Array] = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: jnp.dtype | None = None,
        *,
        key: Key,
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
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            dtype=dtype,
            key=key,
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


class AbstractVectorField(eqx.Module, strict=True):
    @abstractmethod
    def __call__(self, t: ScalarLike, x: Array, args: PyTree) -> Array:
        raise NotImplementedError


class MLPVectorField(AbstractVectorField, strict=True):
    state_size: int
    mlp: eqx.nn.MLP

    def __init__(
        self,
        state_size: int,
        width_size: int,
        depth: int,
        activation: Callable[[Array], Array] = jnn.softplus,
        final_activation: Callable[[Array], Array] = jnn.tanh,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: jnp.dtype | None = None,
        *,
        key: Key,
    ) -> None:
        self.state_size = state_size

        self.mlp = eqx.nn.MLP(
            in_size=state_size + 1,  # Add time
            out_size=state_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            dtype=dtype,
            key=key,
        )

    def __call__(
        self, t: ScalarLike, z: Float[Array, " n"], args: PyTree
    ) -> Float[Array, " n"]:
        input = jnp.concatenate([jnp.expand_dims(t, axis=-1), z])
        return self.mlp(input)


class AbstractTensorVectorField(AbstractVectorField, strict=True):
    @abstractmethod
    def __call__(
        self, t: ScalarLike, z: Float[Array, " state_size"], args: PyTree
    ) -> Float[Array, " state_size input_size"]:
        raise NotImplementedError


class MLPTensorVectorField(AbstractTensorVectorField, strict=True):
    """Matrix vector field for a controlled differential equation."""

    state_size: int
    mlp: TensorMLP

    def __init__(
        self,
        input_size: int,
        state_size: int,
        width_size: int,
        depth: int,
        activation: Callable[[Array], Array] = jnn.softplus,
        final_activation: Callable[[Array], Array] = jnn.tanh,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: jnp.dtype | None = None,
        *,
        key: Key,
    ) -> None:
        self.state_size = state_size

        self.mlp = TensorMLP(
            in_shape=(state_size + 1,),  # Add time
            out_shape=(state_size, input_size + 1),
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            dtype=dtype,
            key=key,
        )

    def __call__(
        self, t: ScalarLike, z: Float[Array, " state_size"], args: PyTree
    ) -> Float[Array, " state_size input_size"]:
        input = jnp.concatenate([jnp.expand_dims(t, axis=-1), z])
        return self.mlp(input)


class NeuralCDE(eqx.Module):
    """Neural controlled differential equations model.

    Maps sequences of inputs to outputs at any time with a hidden latent space.

    Contains three networks:
    - zeta^1: Initial network
      Receives the first time and input pair and outputs the initial state.
    - f: Vector field
      Computes the rate of change of the state with respect to the rate of change of
      the input.
    - zeta^2: Output network
      Compute the output at a given time from the state.
    """

    max_steps: int
    state_index: eqx.nn.StateIndex[
        tuple[
            Float[Array, " num_steps"],
            Float[Array, " num_steps input_size"],
            Float[Array, " num_steps state_size"],
        ]
    ]

    input_size: int
    state_size: int
    output_size: int | Literal["scalar"]

    initial_network: eqx.nn.MLP
    field: AbstractTensorVectorField
    output_network: eqx.nn.MLP

    solver: ClassVar[type[diffrax.AbstractSolver]] = diffrax.Heun

    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int | Literal["scalar"],
        width_size: int,
        depth: int,
        max_steps: int = 1024,
        *,
        key: Key,
        initial_width_size: int | None = None,
        initial_depth: int | None = None,
        field_width_size: int | None = None,
        field_depth: int | None = None,
        output_width_size: int | None = None,
        output_depth: int | None = None,
        field_activation: Callable[[Array], Array] = jnn.softplus,
        field_final_activation: Callable[[Array], Array] = jnn.tanh,
        initial_activation: Callable[[Array], Array] = jnn.relu,
        initial_final_activation: Callable[[Array], Array] = lambda x: x,
        output_activation: Callable[[Array], Array] = jnn.relu,
        output_final_activation: Callable[[Array], Array] = lambda x: x,
    ) -> None:
        self.max_steps = max_steps

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size

        initial_key, field_key, output_key = jr.split(key, 3)

        self.initial_network = eqx.nn.MLP(
            in_size=input_size + 1,
            out_size=state_size,
            width_size=(
                initial_width_size if initial_width_size is not None else width_size
            ),
            depth=initial_depth if initial_depth is not None else 0,
            activation=initial_activation,
            final_activation=initial_final_activation,
            key=initial_key,
        )

        self.field = MLPTensorVectorField(
            input_size=input_size,
            state_size=state_size,
            width_size=field_width_size if field_width_size is not None else width_size,
            depth=field_depth if field_depth is not None else depth,
            activation=field_activation,
            final_activation=field_final_activation,
            key=field_key,
        )

        self.output_network = eqx.nn.MLP(
            in_size=state_size,
            out_size=output_size,
            width_size=(
                output_width_size if output_width_size is not None else width_size
            ),
            depth=output_depth if output_depth is not None else depth,
            activation=output_activation,
            final_activation=output_final_activation,
            key=output_key,
        )

        self.state_index = eqx.nn.StateIndex(self._empty_state())

    def _empty_state(
        self,
    ) -> tuple[
        Float[Array, " num_steps"],
        Float[Array, " num_steps input_size"],
        Float[Array, " num_steps state_size"],
    ]:
        """Get the contents of a new empty state."""
        times = jnp.full((self.max_steps,), jnp.nan)
        inputs = jnp.full((self.max_steps, self.input_size), jnp.nan)
        states = jnp.full((self.max_steps, self.state_size), jnp.nan)
        return times, inputs, states

    def empty_state(self, state: eqx.nn.State) -> eqx.nn.State:
        """Get a new empty state."""
        state = state.set(self.state_index, self._empty_state())
        return state

    @staticmethod
    def latest_index(ts: Float[Array, " num_steps"]) -> Int[ArrayLike, ""]:
        return lax.cond(
            jnp.isnan(ts[0]),
            lambda: -1,
            lambda: jnp.nanargmax(ts),
        )

    def next_state(
        self, ti: Float[Array, ""], xi: Float[Array, " input_size"], state: eqx.nn.State
    ) -> tuple[
        Float[Array, " num_steps"],
        Float[Array, " num_steps input_size"],
        Float[Array, " num_steps state_size"],
    ]:
        """Add new time and input pair to the state."""
        ts, xs, zs = state.get(self.state_index)
        latest_index = self.latest_index(ts)
        ts = eqx.error_if(
            ts,
            ti <= ts[latest_index],
            "new input and time must be later than all previous",
        )

        def shift() -> tuple[
            Float[Array, " num_steps"],
            Float[Array, " num_steps input_size"],
            Float[Array, " num_steps state_size"],
        ]:
            """Shift the saved times and inputs to make room for the new pair."""
            return (
                jnp.roll(ts, -1).at[-1].set(ti),
                jnp.roll(xs, -1, axis=0).at[-1].set(xi),
                jnp.roll(zs, -1, axis=0).at[-1].set(jnp.nan),
            )

        def insert() -> tuple[
            Float[Array, " num_steps"],
            Float[Array, " num_steps input_size"],
            Float[Array, " num_steps state_size"],
        ]:
            """Insert the new time and input pair at the end of the saved times and inputs."""
            return ts.at[latest_index + 1].set(ti), xs.at[latest_index + 1].set(xi), zs

        ts, xs, zs = lax.cond(
            latest_index == self.max_steps - 1,
            shift,
            insert,
        )

        return ts, xs, zs

    def states_from_sequence(
        self,
        ts: Float[Array, " num_steps"],
        xs: Float[Array, " num_steps input_size"],
        save_all: bool = False,
    ) -> Float[Array, " *num_steps state_size"]:
        z0 = self.initial_network(
            jnp.concatenate([jnp.expand_dims(ts[0], axis=-1), xs[0]])
        )

        # jax.debug.print("t0={}, x0={}, z0={}", ts[0], xs[0], z0, ordered=True)
        z0 = eqx.error_if(z0, jnp.any(jnp.isnan(z0)), "NaN in inital state.")

        # Setup control term
        xs = jnp.concatenate([ts[:, None], xs], axis=1)  # Add time to input
        coeffs = diffrax.backward_hermite_coefficients(ts, xs)
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.field, control).to_ode()

        # Setup solve params
        t1 = jnp.nanmax(ts)
        solver = self.solver()
        if isinstance(solver, diffrax.AbstractAdaptiveSolver):
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        if save_all:
            save_at = diffrax.SaveAt(ts=ts)
        else:
            save_at = diffrax.SaveAt(t1=True)

        solution = diffrax.diffeqsolve(
            term,
            solver=solver,
            t0=ts[0],
            t1=t1,
            dt0=None,
            y0=z0,
            stepsize_controller=stepsize_controller,
            saveat=save_at,
        )
        zs = solution.ys
        assert zs is not None
        if save_all:
            # Set zs to nan where ts is nan
            zs = jnp.asarray(jnp.where(jnp.isnan(ts[:, None]), jnp.nan, zs))
            return zs
        else:
            z1 = zs[0]
            return z1

    def __call__(
        self, ti: Float[Array, ""], xi: Float[Array, " input_size"], state: eqx.nn.State
    ) -> tuple[Float[Array, " num_steps state_size"], eqx.nn.State]:
        """Given a new input and time return the output."""
        ts, xs, zs = self.next_state(ti, xi, state)
        z1 = self.states_from_sequence(ts, xs)
        y1 = self.output_network(z1)

        zs = zs.at[self.latest_index(ts)].set(z1)
        state = state.set(self.state_index, (ts, xs, zs))

        return y1, state

    def t1(self, state: eqx.nn.State) -> Float[Array, ""]:
        ts, _, _ = state.get(self.state_index)
        return jnp.nanmax(ts)

    def ts(self, state: eqx.nn.State) -> Float[Array, " num_steps"]:
        ts, _, _ = state.get(self.state_index)
        return ts

    def x1(self, state: eqx.nn.State) -> Float[Array, " input_size"]:
        ts, xs, _ = state.get(self.state_index)
        return xs[self.latest_index(ts)]

    def xs(self, state: eqx.nn.State) -> Float[Array, " num_steps input_size"]:
        _, xs, _ = state.get(self.state_index)
        return xs

    def z1(
        self, state: eqx.nn.State, use_cached: bool = True
    ) -> Float[Array, " state_size"]:
        """Get the latest hidden state of the model."""
        ts, xs, zs = state.get(self.state_index)

        if use_cached:
            z1 = zs[self.latest_index(ts)]
        else:
            z1 = self.states_from_sequence(ts, xs)

        return z1

    def zs(
        self, state: eqx.nn.State, use_cached: bool = True
    ) -> Float[Array, " num_steps state_size"]:
        """Get the hidden state of the model at all times."""
        ts, xs, zs = state.get(self.state_index)

        if not use_cached:
            zs = self.states_from_sequence(ts, xs, save_all=True)

        return zs

    def y1(
        self, state: eqx.nn.State, use_cached: bool = False
    ) -> Float[Array, " output_size"]:
        """Get the final output of of the saved sequence"""
        z1 = self.z1(state, use_cached=use_cached)
        y1 = self.output_network(z1)

        return y1

    def ys(
        self, state: eqx.nn.State, use_cached: bool = False
    ) -> Float[Array, " num_steps output_size"]:
        """Get the output at all times in the saved seuence."""
        zs = self.zs(state, use_cached=use_cached)
        ys = jax.vmap(self.output_network)(zs)

        return ys


if __name__ == "__main__":
    from typing import Any
    import jax.scipy as jsp
    import numpy as np
    import optax
    from matplotlib import pyplot as plt
    from jax import lax

    def random_matrix(n: int, min_eig: float, max_eig: float, *, key: Any) -> Any:
        key, eig_key, matrix_key = jr.split(key, 3)
        eigvals = jr.uniform(eig_key, (n,), minval=min_eig, maxval=max_eig)
        rand_mat = jr.normal(matrix_key, (n, n))
        Q, R = jnp.linalg.qr(rand_mat)
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
        key: Any,
    ):
        initial_key, matrix_key, noise_key = jr.split(key, 3)
        ts = jnp.linspace(t0, t1, length)
        x0 = jr.uniform(initial_key, (n,))
        matrix = random_matrix(n, -1, 1e-1, key=matrix_key)
        xs = jax.vmap(lambda t: jsp.linalg.expm(t * matrix) @ x0)(ts)
        if add_noise:
            xs = xs + jr.normal(noise_key, xs.shape) * noise_scale
        y1 = jnp.diag(matrix)
        return xs, ts, y1

    def simulate_trajectory(neural_cde, ts, xs, init_state):
        def scan_fn(carry, inputs):
            t, x = inputs
            y, new_state = neural_cde(t, x, carry)
            return new_state, y

        final_state, _ = lax.scan(scan_fn, init_state, (ts, xs))
        y_pred = neural_cde.y1(final_state)
        return y_pred

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss_grad(neural_cde, state, xs_batch, ts_batch, y_batch):
        init_state = neural_cde.empty_state(state)
        y_preds = jax.vmap(
            lambda ts, xs: simulate_trajectory(neural_cde, ts, xs, init_state)
        )(ts_batch, xs_batch)
        return jnp.mean((y_preds - y_batch) ** 2)

    dataset_size = 128
    data_length = 128
    data_time = 4 * jnp.pi
    add_noise = True
    data_dim = 2
    learning_rate = 1e-4
    epochs = 128
    batch_size = 16
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

    neural_cde, state = eqx.nn.make_with_state(NeuralCDE)(
        input_size=data_dim,
        state_size=2,
        output_size=data_dim,
        width_size=64,
        depth=2,
        max_steps=data_length,
        field_activation=jnn.relu,
        key=model_key,
    )
    schedule = optax.cosine_decay_schedule(
        learning_rate, decay_steps=(dataset_size // batch_size) * epochs
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(neural_cde, eqx.is_array))

    num_batches = dataset_size // batch_size
    model_dynamic, model_static = eqx.partition(neural_cde, eqx.is_inexact_array)

    def batch_scan(carry, inputs):
        model_dynamic, opt_state, model_state = carry
        model = eqx.combine(model_dynamic, model_static)
        xs_batch, ts_batch, y_batch = inputs
        loss, grads = loss_grad(model, model_state, xs_batch, ts_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        model_dynamic, _ = eqx.partition(model, eqx.is_inexact_array)
        return (model_dynamic, opt_state, model_state), loss

    def epoch_scan(carry, epoch_idx):
        key, model_dynamic, opt_state, model_state = carry
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, dataset_size)
        indices = perm.reshape(num_batches, -1)
        (model_dynamic, opt_state, model_state), epoch_losses = jax.lax.scan(
            batch_scan,
            (model_dynamic, opt_state, model_state),
            (xs[indices], ts[indices], y1[indices]),
        )
        epoch_loss = jnp.mean(epoch_losses)
        jax.debug.print("Epoch {:03}: loss={:.5f}", epoch_idx, epoch_loss, ordered=True)
        return (key, model_dynamic, opt_state, model_state), epoch_loss

    init_carry = (key, model_dynamic, opt_state, state)
    (final_key, final_model_dynamic, final_opt_state, final_state), epoch_losses = (
        lax.scan(epoch_scan, init_carry, jnp.arange(epochs))
    )
    print(f"Final training loss: {epoch_losses[-1]:.5f}")

    test_key = jr.split(final_key)[0]
    t0_test = 0.0
    t1_test = t0_test + data_time
    test_xs, test_ts, test_y1 = get_data(
        data_dim,
        data_length,
        t0=t0_test,
        t1=t1_test,
        add_noise=False,
        key=test_key,
    )
    test_loss, _ = loss_grad(
        eqx.combine(final_model_dynamic, model_static),
        final_state,
        test_xs[None, ...],
        test_ts[None, ...],
        test_y1[None, ...],
    )
    print(f"Test loss: {test_loss:.5f}")

    test_zs = neural_cde.states_from_sequence(test_ts, test_xs, save_all=True)
    test_ys = jax.vmap(neural_cde.output_network)(test_zs)

    test_zs = np.array(test_zs)
    test_ys = np.array(test_ys)
    test_y1 = np.array(test_y1)

    plt.figure()
    plt.plot(test_zs[:, 0], test_zs[:, 1], label="Hidden State Trajectory")
    plt.plot(test_ys[:, 0], test_ys[:, 1], "--", label="Output Trajectory")
    plt.scatter(
        test_y1[0], test_y1[1], color="red", marker="o", s=100, label="Target Y"
    )
    plt.xlabel("Dimension 0")
    plt.ylabel("Dimension 1")
    plt.title("State-space Trajectories and Target Y")
    plt.legend()
    plt.show()
