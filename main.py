"""ODE RNN PPO

An implementation of Proximal Policy Optimization with a Neural Controlled Differential Equations.
"""

import dataclasses
import logging
from functools import partial
from typing import Any, Callable, Literal

import chex
import diffrax
import equinox as eqx
import gymnax as gym
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax
from gymnax import wrappers
from gymnax.environments import environment, spaces
from gymnax.visualize import Visualizer
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key, PyTree
from torch.utils.tensorboard.writer import SummaryWriter

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 64-bit precision required for numerical stability
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

    def __call__(self, t: Array, x: Array, args) -> Array:
        """Time should be added as an extra dimension to the input instead of used here."""
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
        field_activation: Callable[[Array], Array] = jnn.tanh,
        output_final_activation: Callable[[Array], Array] = jnn.tanh,
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
        - field_activation: The activation function for the field.
        - output_final_activation: The activation function for the output.
        """
        initial_key, field_key, output_key = jr.split(key, 3)

        self.input_size = input_size
        self.state_size = hidden_size
        self.output_size = output_size

        if input_size == "scalar":
            input_size = 1

        # Default to a linear initial condition
        self.initial = eqx.nn.MLP(
            in_size=input_size + 1,
            out_size=hidden_size,
            width_size=(
                initial_width_size if initial_width_size is not None else width_size
            ),
            depth=initial_depth if initial_depth is not None else 0,
            key=initial_key,
        )

        # Matrix output of hidden_size x (input_size + 1)
        # Tanh to prevent blowing up
        self.field = Field(
            in_shape=(hidden_size,),
            out_shape=(hidden_size, input_size + 1),
            width_size=width_size,
            depth=depth,
            activation=field_activation,
            final_activation=jnn.tanh,
            key=field_key,
        )

        # Default to a linear output
        self.output = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=output_size,
            width_size=(
                output_width_size if output_width_size is not None else width_size
            ),
            depth=output_depth if output_depth is not None else depth,
            final_activation=output_final_activation,
            key=output_key,
        )

    def __call__(
        self,
        ts: Float[Array, " N"],
        xs: Float[Array, " N input_size"],
        *,
        z0: Float[Array, " state_size"],
        t1: Int[Array, ""] | None = None,
        evolving_out: bool = False,
    ) -> tuple[Array, Array]:
        """Compute the output of the Neural CDE.

        Evaluates the Neural CDE at a series of times and inputs using cubic
        hermite splines with backward differences.

        Arguments:
        - ts: The times to evaluate the Neural CDE.
        - xs: The inputs to the Neural CDE.
        - z0: The initial state of the Neural CDE.
        - t1: The final time to evaluate the Neural CDE.
        - evolving_out: If True, return the output at each time step.

        Returns:
        - z1: The final state of the Neural CDE.
        - y1: The output of the Neural CDE.
        """
        assert ts.ndim == 1
        assert xs.ndim == 2
        assert z0.ndim == 1
        assert xs.shape[0] == ts.shape[0]

        if t1 is not None:
            assert t1.ndim == 0
            t1 = eqx.error_if(t1, jnp.isnan(t1), "t1 is NaN")
        if evolving_out:
            ts = eqx.error_if(ts, jnp.isnan(ts).any(), "ts contains NaNs")
            xs = eqx.error_if(xs, jnp.isnan(xs).any(), "xs contains NaNs")

        if self.input_size == "scalar":
            assert xs.ndim == 1
            xs = jnp.expand_dims(xs, axis=-1)

        # Don't actually solve an ODE if there's no time change
        if len(ts) == 1:
            if evolving_out:
                zs = jnp.expand_dims(z0, axis=0)
                return zs, jax.vmap(self.output)(zs)
            else:
                z1 = z0
                return z1, self.output(z1)

        # Create a control term with a cubic interpolation of the input
        xs = jnp.concatenate([ts[:, None], xs], axis=1)  # Add time to input
        coeffs = diffrax.backward_hermite_coefficients(ts, xs)
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.field, control).to_ode()  # pyright: ignore

        solver = diffrax.Tsit5()
        if isinstance(solver, diffrax.AbstractAdaptiveSolver):
            dt0 = None
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        else:
            dt0 = jnp.nanmin(ts[1:] - ts[:-1])
            stepsize_controller = diffrax.ConstantStepSize()

        if evolving_out:
            t1 = jnp.max(ts)
            saveat = diffrax.SaveAt(ts=ts)
        else:
            if t1 is None:
                t1 = jnp.max(ts)
            saveat = diffrax.SaveAt(t1=True)

        solution = diffrax.diffeqsolve(
            term,
            solver=solver,
            t0=ts[0],
            t1=t1,
            dt0=dt0,
            y0=z0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )
        zs = solution.ys
        assert zs is not None

        if evolving_out:
            return zs, jax.vmap(self.output)(zs)
        else:
            z1 = zs[-1]
            return z1, self.output(z1)

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


class CDEAgent(eqx.Module):
    input_size: int
    state_size: int
    action_size: int
    actor: NeuralCDE
    action_logstd: Float[Array, " action_size"]
    critic: eqx.nn.MLP

    action_space: spaces.Box
    const_std: bool

    def __init__(
        self,
        env: environment.Environment | GymnaxWrapper,
        env_params: gym.EnvParams,
        hidden_size: int,
        width_size: int,
        depth: int,
        *,
        key: Key,
        initial_width_size: int | None = None,
        initial_depth: int | None = None,
        output_width_size: int | None = None,
        output_depth: int | None = None,
        actor_width_size: int | None = None,
        actor_depth: int | None = None,
        critic_width_size: int | None = None,
        critic_depth: int | None = None,
        field_activation: Callable[[Array], Array] = jnn.softplus,
        actor_output_final_activation: Callable[[Array], Array] = jnn.tanh,
        const_std: bool = False,
    ) -> None:
        """Create an actor critic model with a neural CDE.

        Uses the neural CDE to process inputs and generate a state
        that is shared between the actor and critic.

        Arguments:
        - env: Environment to interact with.
        - env_params: Parameters for the environment.
        - hidden_size: Size of the hidden state in the neural CDE.
        - width_size: Width of the neural CDE.
        - depth: Depth of the neural CDE.
        - key: Random key for initialization.
        - initial_width_size: Width of the initialization network for the neural CDE state.
        - initial_depth: Depth of the initialization network for the neural CDE state.
        - output_width_size: Width of the output network for the neural CDE state.
        - output_depth: Depth of the output network for the neural CDE state.
        - actor_width_size: Width of the actor network.
        - actor_depth: Depth of the actor network.
        - critic_width_size: Width of the critic network.
        - critic_depth: Depth of the critic network.
        - field_activation: Activation function for the field in the neural CDE.
        - const_std: Whether to use a constant standard deviation for the action.
        """
        assert isinstance(env.action_space(env_params), spaces.Box)
        assert isinstance(env.observation_space(env_params), spaces.Box)

        actor_key, actor_weight_key, critic_key, critic_weight_key = jr.split(key, 4)

        self.input_size = int(
            jnp.asarray(env.observation_space(env_params).shape).prod()
        )
        self.state_size = hidden_size
        self.action_size = int(jnp.asarray(env.action_space(env_params).shape).prod())

        self.action_space = env.action_space(env_params)

        self.actor = NeuralCDE(
            input_size=self.input_size,
            hidden_size=hidden_size,
            output_size=self.action_size,
            width_size=actor_width_size if actor_width_size is not None else width_size,
            depth=actor_depth if actor_depth is not None else depth,
            initial_width_size=initial_width_size,
            initial_depth=initial_depth,
            output_width_size=output_width_size,
            output_depth=output_depth,
            key=actor_key,
            field_activation=field_activation,
            output_final_activation=actor_output_final_activation,
        )

        self.actor = eqx.tree_at(
            lambda ncde: ncde.output.layers[-1].weight,
            self.actor,
            replace_fn=lambda x: jr.normal(actor_weight_key, x.shape) * 0.01,
        )

        self.action_logstd = jnp.zeros(self.action_size)
        self.const_std = const_std

        self.critic = eqx.nn.MLP(
            in_size=self.input_size,
            out_size="scalar",
            width_size=(
                critic_width_size if critic_width_size is not None else width_size
            ),
            depth=critic_depth if critic_depth is not None else depth,
            key=critic_key,
        )

        self.critic = eqx.tree_at(
            lambda mlp: mlp.layers[-1].weight,
            self.critic,
            replace_fn=lambda x: jr.normal(critic_weight_key, x.shape) * 1,
        )

    def get_action_and_value(
        self,
        ts: Float[Array, " N"],
        xs: Float[Array, " N {self.input_size}"],
        a1: Float[Array, " {self.action_size}"] | None = None,
        z0: Float[Array, " {self.state_size}"] | None = None,
        t1: Float[Array, ""] | None = None,
        *,
        key: Key | None = None,
        evolving_out: bool = False,
    ) -> tuple[
        Float[Array, " *N {self.state_size}"],
        Float[Array, " *N {self.action_size}"],
        Float[Array, " *N"],
        Float[Array, " *N"],
        Float[Array, " *N"],
    ]:
        """Return a final action and value for the given inputs.

        If an action is provided, it assumed to be fixed and the
        value is computed for the given action. Otherwise, the action is
        computed from the given inputs and sampled from the action distribution
        of the actor-critic model over the inputs.

        Arguments:
        - ts: Time steps of the inputs.
        - xs: Inputs to the model.
        - a1: Optional fixed action.
        - z0: Optional initial state.
        - t1: Optional final time.
        - key: Key for sampling the action.
        - evolving_out: Whether to compute the output for every time step.

        Returns:
        - z1: Final state of the actor-critic model.
        - a1: Action computed from the inputs.
        - log prob: Log probability of the action.
        - entropy: Entropy of the action distribution.
        - value: Value of the final state.
        """
        z0 = z0 or self.initial_state(ts[0], xs[0])
        t1 = t1 or jnp.nanmax(ts)

        z1, action_mean = self.actor(ts, xs, z0=z0, t1=t1, evolving_out=evolving_out)
        action_mean = (action_mean + 1) * (
            self.action_space.high - self.action_space.low
        ) / 2 + self.action_space.low

        if self.const_std:
            action_std = jnp.ones_like(action_mean)
        else:
            action_std = jnp.exp(self.action_logstd)

        if evolving_out:
            value = jax.vmap(self.critic)(xs)
        else:
            last_valid = jnp.sum(~jnp.isnan(ts)) - 1
            value = self.critic(xs[last_valid])

        if a1 is None:
            if key is not None:
                a1 = jr.normal(key, action_mean.shape) * action_std + action_mean
            else:
                a1 = action_mean
        else:
            a1 = jnp.asarray(a1)

        log_probs = jax.scipy.stats.norm.logpdf(a1, action_mean, action_std)
        entropies = 0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1)

        if evolving_out:
            log_prob = jax.vmap(jnp.sum)(log_probs)
            entropy = jax.vmap(jnp.sum)(entropies)
        else:
            log_prob = jnp.sum(log_probs)
            entropy = jnp.sum(entropies)

        if evolving_out:
            z1 = eqx.error_if(
                z1, z1.shape != (ts.shape[0], self.state_size), "z1 has incorrect size"
            )
            a1 = eqx.error_if(
                a1, a1.shape != (ts.shape[0], self.action_size), "a1 has incorrect size"
            )
        else:
            z1 = eqx.error_if(
                z1, z1.shape != (self.state_size,), "z1 has incorrect size"
            )
            a1 = eqx.error_if(
                a1, a1.shape != (self.action_size,), "a1 has incorrect size"
            )

        return z1, a1, log_prob, entropy, value

    def initial_state(
        self,
        t0: Float[ArrayLike, ""],
        x0: Float[ArrayLike, " {self.input_size}"],
    ) -> Float[Array, " {self.state_size}"]:
        """Generate an initial state z0 from an inital input x0 and time t0.

        Arguments:
        - t0: Time of the initial input.
        - x0: Initial input.

        Returns:
        - z0: Initial state of the actor-critic model.
        """
        t0 = jnp.asarray(t0)
        x0 = jnp.asarray(x0)
        return self.actor.initial_state(t0, x0)


class MLPAgent(eqx.Module):
    """Agent with the same interface as the stateful CDE Agent but uses regular MLPs
    for action and value generation.
    """

    input_size: int
    state_size: int
    action_size: int
    actor: eqx.nn.MLP
    action_logstd: Float[Array, " action_size"]
    critic: eqx.nn.MLP

    def __init__(
        self,
        env: environment.Environment | GymnaxWrapper,
        env_params: gym.EnvParams,
        width_size: int,
        depth: int,
        *,
        key: Key,
        actor_width_size: int | None = None,
        actor_depth: int | None = None,
        critic_width_size: int | None = None,
        critic_depth: int | None = None,
    ) -> None:
        assert isinstance(env.action_space(env_params), spaces.Box)
        assert isinstance(env.observation_space(env_params), spaces.Box)

        actor_key, actor_weight_key, critic_key, critic_weight_key = jr.split(key, 4)

        self.input_size = int(
            jnp.asarray(env.observation_space(env_params).shape).prod()
        )
        self.state_size = 1
        self.action_size = int(jnp.asarray(env.action_space(env_params).shape).prod())

        self.actor = eqx.nn.MLP(
            in_size=self.input_size,
            out_size=self.action_size,
            width_size=actor_width_size if actor_width_size is not None else width_size,
            depth=actor_depth if actor_depth is not None else depth,
            key=actor_key,
        )

        self.actor = eqx.tree_at(
            lambda mlp: mlp.layers[-1].weight,
            self.actor,
            replace_fn=lambda x: jr.normal(actor_weight_key, x.shape) * 0.01,
        )

        self.action_logstd = jnp.zeros(self.action_size)

        self.critic = eqx.nn.MLP(
            in_size=self.input_size,
            out_size="scalar",
            width_size=(
                critic_width_size if critic_width_size is not None else width_size
            ),
            depth=critic_depth if critic_depth is not None else depth,
            key=critic_key,
        )

        self.critic = eqx.tree_at(
            lambda mlp: mlp.layers[-1].weight,
            self.critic,
            replace_fn=lambda x: jr.normal(critic_weight_key, x.shape) * 1,
        )

    def get_action_and_value(
        self,
        ts: Float[Array, " N"],
        xs: Float[Array, " N {self.input_size}"],
        a1: Float[Array, " {self.action_size}"] | None = None,
        z0: Float[Array, " {self.state_size}"] | None = None,
        t1: Float[Array, ""] | None = None,
        *,
        key: Key | None = None,
        evolving_out: bool = False,
    ) -> tuple[
        Float[Array, " *N {self.state_size}"],
        Float[Array, " *N {self.action_size}"],
        Float[Array, " *N"],
        Float[Array, " *N"],
        Float[Array, " *N"],
    ]:
        if evolving_out:
            action_mean = jax.vmap(self.actor)(xs)
            value = jax.vmap(self.critic)(xs)
            z1 = jnp.zeros((ts.shape[0], self.state_size))
        else:
            last_valid = jnp.sum(~jnp.isnan(ts)) - 1
            action_mean = self.actor(xs[last_valid])
            value = self.critic(xs[last_valid])
            z1 = jnp.zeros((self.state_size,))

        # action_std = jnp.exp(self.action_std)
        action_std = jnp.ones_like(action_mean)

        if a1 is None:
            if key is not None:
                a1 = jr.normal(key, action_mean.shape) * action_std + action_mean
            else:
                a1 = action_mean
        else:
            a1 = jnp.asarray(a1)

        log_probs = jax.scipy.stats.norm.logpdf(a1, action_mean, action_std)
        entropies = 0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1)

        if evolving_out:
            log_prob = jax.vmap(jnp.sum)(log_probs)
            entropy = jax.vmap(jnp.sum)(entropies)
        else:
            log_prob = jnp.sum(log_probs)
            entropy = jnp.sum(entropies)

        if evolving_out:
            z1 = eqx.error_if(
                z1, z1.shape != (ts.shape[0], self.state_size), "z1 has incorrect size"
            )
            a1 = eqx.error_if(
                a1, a1.shape != (ts.shape[0], self.action_size), "a1 has incorrect size"
            )
        else:
            z1 = eqx.error_if(
                z1, z1.shape != (self.state_size,), "z1 has incorrect size"
            )
            a1 = eqx.error_if(
                a1, a1.shape != (self.action_size,), "a1 has incorrect size"
            )

        return z1, a1, log_prob, entropy, value

    def initial_state(
        self,
        t0: Float[ArrayLike, ""],
        x0: Float[ArrayLike, " {self.input_size}"],
    ) -> Float[Array, " {self.state_size}"]:
        return jnp.zeros(self.state_size)


@chex.dataclass
class EpisodeState:
    step: Int[Array, ""]
    env_state: gym.EnvState
    observations: Float[Array, " num_steps observation_size"]
    times: Float[Array, " num_steps"]


@chex.dataclass
class RolloutBuffer:
    observations: Float[Array, " num_steps observation_size"]
    actions: Float[Array, " num_steps action_size"]
    log_probs: Float[Array, " num_steps"]
    entropies: Float[Array, " num_steps"]
    values: Float[Array, " num_steps"]
    rewards: Float[Array, " num_steps"]
    terminations: Bool[Array, " num_steps"]
    truncations: Bool[Array, " num_steps"]
    advantages: Float[Array, " num_steps"]
    returns: Float[Array, " num_steps"]
    times: Float[Array, " num_steps"]


@chex.dataclass
class TrainingState:
    opt_state: optax.OptState
    global_step: Int[Array, ""]


@chex.dataclass
class PPOStats:
    total_loss: Float[Array, " *N"]
    policy_loss: Float[Array, " *N"]
    value_loss: Float[Array, " *N"]
    entropy_loss: Float[Array, " *N"]
    state_loss: Float[Array, " *N"]
    approx_kl: Float[Array, " *N"]
    variance: Float[Array, " *N"]
    explained_variance: Float[Array, " *N"]
    grad_norm: Float[Array, " *N"]
    update_norm: Float[Array, " *N"]
    learning_rate: Float[Array, " *N"]


@chex.dataclass
class PPOArguments:
    run_name: str

    num_iterations: int
    num_steps: int = 1024
    num_epochs: int = 16
    num_minibatches: int = 32
    minibatch_size: int = 32
    num_batches: int = 8
    batch_size: int = 4

    agent_timestep: float = 1.0

    gamma: float = 0.99
    gae_lambda: float = 0.95

    learning_rate: float = 1e-3
    anneal_learning_rate: bool = True

    normalize_advantage: bool = False
    clip_coefficient: float = 0.2
    clip_value_loss: bool = True
    entropy_coefficient: float = 0.0
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    state_coefficient: float = 0.1
    target_kl: float | None = None

    tb_logging: bool = True


def make_empty(cls, **kwargs):
    """Create an empty dataclass with the fields set to None.

    If a field is provided in kwargs, it will be set to the value in kwargs.
    """
    dcls = cls(**{f.name: None for f in dataclasses.fields(cls)})
    for k, v in kwargs.items():
        dcls.__setattr__(k, v)

    return dcls


def reset_episode(
    env: environment.Environment | GymnaxWrapper,
    env_params: gym.EnvParams,
    args: PPOArguments,
    *,
    key: Key,
) -> EpisodeState:
    """Reset the environment and agent to a fresh state.

    Initializes the time to a random value between 0 and 100.

    Arguments:
    - env: Environment to reset.
    - env_params: Parameters for the environment.
    - agent: Agent to reset.
    - args: PPO arguments.
    - key: Random key for resetting the environment.

    Returns:
    - training_state: A fresh training state.
    """
    reset_key, time_key = jr.split(key, 2)
    observation, env_state = env.reset(reset_key, env_params)
    observations = jnp.full((args.num_steps, observation.shape[0]), jnp.nan)
    observations = observations.at[0].set(observation)

    agent_times = jnp.full((args.num_steps,), jnp.nan)
    agent_times = agent_times.at[0].set(
        jr.uniform(time_key, (), minval=0.0, maxval=100.0)
    )

    # Fix the LogWrapper typing
    if isinstance(env, wrappers.LogWrapper):
        env_state = eqx.tree_at(
            lambda s: s.episode_returns,
            env_state,
            replace_fn=lambda x: jnp.astype(x, jnp.float64),
        )
        env_state = eqx.tree_at(
            lambda s: s.returned_episode_returns,
            env_state,
            replace_fn=lambda x: jnp.astype(x, jnp.float64),
        )

    return EpisodeState(
        step=jnp.array(0),
        env_state=env_state,
        observations=observations,
        times=agent_times,
    )


def rollover_episode_state(episode_state: EpisodeState) -> EpisodeState:
    """If an episode has already filled a buffer, rollover the episode state without
    resetting the environment.
    """
    new_step = jnp.array(0)
    new_times = (
        jnp.full_like(episode_state.times, jnp.nan)
        .at[0]
        .set(episode_state.times[episode_state.step])
    )
    new_observations = (
        jnp.full_like(episode_state.observations, jnp.nan)
        .at[0]
        .set(episode_state.observations[episode_state.step])
    )
    return EpisodeState(
        step=new_step,
        env_state=episode_state.env_state,
        observations=new_observations,
        times=new_times,
    )


def env_step(
    env: environment.Environment | GymnaxWrapper,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    episode_state: EpisodeState,
    args: PPOArguments,
    *,
    key: Key,
    writer: SummaryWriter | None = None,
    global_step: int | None = None,
) -> tuple[EpisodeState, RolloutBuffer, dict]:
    """Step the environment and agent forward one step and store the data in a rollout buffer.

    Arguments:
    - env: Environment to interact with.
    - env_params: Parameters for the environment.
    - agent: Agent to interact with the environment.
    - episode_state: Current episode state.
    - args: PPO arguments.
    - key: Random key for sampling.

    Returns:
    - episode_state: New episode state.
    - buffer: Buffer of data from the episode step.
    - info: Information about the episode step.
    """
    actor_key, env_key, reset_key = jr.split(key, 3)

    z1, action, log_prob, entropy, value = agent.get_action_and_value(
        ts=episode_state.times,
        xs=episode_state.observations,
        key=actor_key,
    )

    buffer = make_empty(
        RolloutBuffer,
        observations=episode_state.observations[episode_state.step],
        actions=action,
        times=episode_state.times[episode_state.step],
    )

    clipped_action = jnp.clip(
        action, env.action_space(env_params).low, env.action_space(env_params).high
    )

    if writer is not None:
        jax.debug.callback(
            lambda action, global_step: writer.add_scalar(  # pyright: ignore
                "episode/action", float(action), int(global_step)
            ),
            action[0],
            global_step,
        )
        jax.debug.callback(
            lambda action, global_step: writer.add_scalar(  # pyright: ignore
                "episode/clipped_action", float(action), int(global_step)
            ),
            clipped_action[0],
            global_step,
        )
        jax.debug.callback(
            lambda z1, global_step: writer.add_scalar(  # pyright: ignore
                "episode/z1", float(z1), int(global_step)
            ),
            jnp.linalg.norm(z1),
            global_step,
        )
    observation, env_state, reward, done, info = env.step(
        env_key, episode_state.env_state, clipped_action, env_params
    )
    termination = done
    truncation = jnp.array(False)

    buffer.rewards = reward
    buffer.terminations = termination
    buffer.truncations = truncation
    buffer.log_probs = log_prob
    buffer.entropies = entropy
    buffer.values = value

    episode_state = EpisodeState(
        step=episode_state.step + 1,
        env_state=env_state,
        observations=episode_state.observations.at[episode_state.step + 1].set(
            observation
        ),
        times=episode_state.times.at[episode_state.step + 1].set(
            episode_state.times[episode_state.step] + args.agent_timestep
        ),
    )

    # If the episode is done, reset the environment.
    episode_state = jax.lax.cond(
        termination | truncation,
        lambda: reset_episode(env, env_params, args, key=reset_key),
        lambda: episode_state,
    )

    return episode_state, buffer, info


def collect_rollout(
    env: environment.Environment | GymnaxWrapper,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    episode_state: EpisodeState,
    training_state: TrainingState,
    args: PPOArguments,
    key: Key,
    writer: SummaryWriter | None = None,
) -> tuple[EpisodeState, TrainingState, RolloutBuffer]:
    def write_episode_stats(info: dict[str, Array], global_step: Array):
        reward = info["returned_episode_returns"]
        length = info["returned_episode_lengths"]
        jax.debug.callback(
            lambda reward, length, global_step: logger.debug(
                f"Episode finished with reward {reward} and length {length} at {global_step}"
            ),
            reward,
            length,
            global_step,
        )
        jax.debug.callback(
            lambda reward, global_step: writer.add_scalar(  # pyright: ignore
                "episode/reward", float(reward), int(global_step)
            ),
            reward,
            global_step,
        )
        jax.debug.callback(
            lambda length, global_step: writer.add_scalar(  # pyright: ignore
                "episode/length", float(length), int(global_step)
            ),
            length,
            global_step,
        )

    episode_state = rollover_episode_state(episode_state)

    def scan_step(
        carry: tuple[EpisodeState, TrainingState, Key],
        _: None,
    ) -> tuple[tuple[EpisodeState, TrainingState, Key], RolloutBuffer]:
        episode_state, training_state, key = carry
        carry_key, step_key = jr.split(key, 2)

        training_state = TrainingState(
            opt_state=training_state.opt_state,
            global_step=training_state.global_step + 1,
        )
        episode_state, buffer, info = env_step(
            env,
            env_params,
            agent,
            episode_state,
            args,
            key=step_key,
            writer=writer,
            global_step=training_state.global_step,
        )

        if writer is not None:
            jax.lax.cond(
                info["returned_episode"],
                lambda: write_episode_stats(info, training_state.global_step),
                lambda: None,
            )

        return (episode_state, training_state, carry_key), buffer

    (episode_state, training_state, _), rollout = jax.lax.scan(
        scan_step, (episode_state, training_state, key), length=args.num_steps
    )

    return episode_state, training_state, rollout


def calculate_gae(
    rewards: Float[Array, " num_steps"],
    values: Float[Array, " num_steps"],
    terminations: Bool[Array, " num_steps"],
    args: PPOArguments,
) -> tuple[Float[Array, " num_steps"], Float[Array, " num_steps"]]:
    def scan_fn(carry: Float, t: Int) -> tuple[Float, Float]:
        gae = carry
        reward = rewards[t]
        value = values[t]
        next_value = jnp.where(t < args.num_steps - 1, values[t + 1], value)
        not_done = 1.0 - terminations[t].astype(jnp.float32)

        delta = reward + args.gamma * next_value * not_done - value
        gae = delta + args.gamma * args.gae_lambda * not_done * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        scan_fn, init=0.0, xs=jnp.arange(args.num_steps - 1, -1, -1)
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


def minibatch_loss(
    agent: CDEAgent,
    times: Float[Array, " minibatch_size"],
    observations: Float[Array, " minibatch_size observation_size"],
    actions: Float[Array, " minibatch_size action_size"],
    log_probs: Float[Array, " minibatch_size"],
    values: Float[Array, " minibatch_size"],
    advantages: Float[Array, " minibatch_size"],
    returns: Float[Array, " minibatch_size"],
    args: PPOArguments,
) -> tuple[Float[Array, ""], PPOStats]:
    zs, _, new_log_probs, entropies, new_values = agent.get_action_and_value(
        ts=times, xs=observations, a1=actions, evolving_out=True
    )

    log_ratio = new_log_probs - log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = jnp.mean(ratio - log_ratio) - 1

    if args.normalize_advantage:
        advantages = (advantages - jnp.mean(advantages)) / (
            jnp.std(advantages) + jnp.finfo(advantages.dtype).eps
        )

    policy_loss = -jnp.mean(
        jnp.minimum(
            advantages * ratio,
            advantages
            * jnp.clip(ratio, 1 - args.clip_coefficient, 1 + args.clip_coefficient),
        )
    )

    if args.clip_value_loss:
        clipped_values = values + jnp.clip(
            new_values - values, -args.clip_coefficient, args.clip_coefficient
        )
        value_loss = (
            jnp.mean(
                jnp.maximum(
                    jnp.square(new_values - returns),
                    jnp.square(clipped_values - returns),
                )
            )
            / 2
        )
    else:
        value_loss = jnp.mean(jnp.square(new_values - returns)) / 2

    entropy_loss = jnp.mean(entropies)

    state_loss = jnp.mean(jnp.linalg.norm(zs, axis=-1))

    loss = (
        policy_loss
        + args.value_coefficient * value_loss
        + args.state_coefficient * state_loss
        - args.entropy_coefficient * entropy_loss
    )
    stats = make_empty(
        PPOStats,
        total_loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        state_loss=state_loss,
        approx_kl=approx_kl,
    )

    return loss, stats


def batch_loss(
    agent: CDEAgent,
    times: Float[Array, " batch_size minibatch_size"],
    observations: Float[Array, " batch_size minibatch_size observation_size"],
    actions: Float[Array, " batch_size minibatch_size action_size"],
    log_probs: Float[Array, " batch_size minibatch_size"],
    values: Float[Array, " batch_size minibatch_size"],
    advantages: Float[Array, " batch_size minibatch_size"],
    returns: Float[Array, " batch_size minibatch_size"],
    args: PPOArguments,
) -> tuple[Float[Array, ""], PPOStats]:
    losses, stats = jax.vmap(partial(minibatch_loss, agent, args=args))(
        times, observations, actions, log_probs, values, advantages, returns
    )
    return jnp.mean(losses), jax.tree.map(jnp.mean, stats)


batch_grad = eqx.filter_value_and_grad(batch_loss, has_aux=True)


def train_on_batch(
    agent: CDEAgent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: RolloutBuffer,
    args: PPOArguments,
) -> tuple[CDEAgent, optax.OptState, PPOStats]:
    (_, stats), grads = batch_grad(
        agent,
        rollout.times,
        rollout.observations,
        rollout.actions,
        rollout.log_probs,
        rollout.values,
        rollout.advantages,
        rollout.returns,
        args,
    )

    stats.learning_rate = opt_state["adam"].hyperparams[  # pyright: ignore
        "learning_rate"
    ]

    updates, opt_state = optimizer.update(grads, opt_state)
    agent = eqx.apply_updates(agent, updates)

    stats.grad_norm = jnp.linalg.norm(
        jnp.concatenate(jax.tree.flatten(jax.tree.map(jnp.ravel, grads))[0])
    )
    stats.update_norm = jnp.linalg.norm(
        jnp.concatenate(jax.tree.flatten(jax.tree.map(jnp.ravel, updates))[0])
    )

    return agent, opt_state, stats


def get_batch_indices(
    batch_size: int, dataset_size: int, num_batches: int, key: Key
) -> Int[Array, " {num_batches} {batch_size}"]:
    indices = jr.permutation(key, dataset_size)
    total_required = batch_size * num_batches
    tiled_indices = jnp.tile(indices, (total_required // dataset_size + 1,))
    selected = tiled_indices[:total_required]
    return selected.reshape((num_batches, batch_size))


def epoch(
    agent: CDEAgent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: RolloutBuffer,
    args: PPOArguments,
    key: Key,
) -> tuple[CDEAgent, optax.OptState, PPOStats]:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)
    minibatch_key, batch_key = jr.split(key, 2)
    rollout = split_into_minibatches(rollout, args, minibatch_key)

    def scan_batch(
        carry: tuple[PyTree, optax.OptState], xs: Int[Array, " batch_size"]
    ) -> tuple[tuple[PyTree, optax.OptState], PPOStats]:
        agent_dynamic, opt_state = carry
        agent = eqx.combine(agent_dynamic, agent_static)

        batch_rollout = jax.tree.map(
            partial(jnp.take, indices=xs, axis=0),
            rollout,
        )

        agent, opt_state, stats = train_on_batch(
            agent,
            optimizer,
            opt_state,
            batch_rollout,
            args,
        )
        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)
        return (agent_dynamic, opt_state), stats

    indices = get_batch_indices(
        args.batch_size, args.num_minibatches, args.num_batches, batch_key
    )
    (agent_dynamic, opt_state), stats = jax.lax.scan(
        scan_batch, (agent_dynamic, opt_state), indices
    )

    stats = jax.tree.map(jnp.mean, stats)
    agent = eqx.combine(agent_dynamic, agent_static)

    return agent, opt_state, stats


def split_into_minibatches(
    rollout: RolloutBuffer, args: PPOArguments, key: Key
) -> RolloutBuffer:
    """Split a rollout into contiguous minibatches that do not contain terminations
    or truncations.
    """
    dones = rollout.terminations | rollout.truncations
    possible_indices = jnp.arange(args.num_steps - args.minibatch_size + 1)
    valid_indices = jax.vmap(
        lambda i: ~jnp.any(lax.dynamic_slice(dones, (i,), (args.minibatch_size,)))
    )(possible_indices)

    sampled_indices = jr.choice(
        key, possible_indices, shape=(args.num_minibatches,), p=valid_indices
    )
    indices = sampled_indices[:, None] + jnp.arange(args.minibatch_size)
    # indices = jr.permutation(key, jnp.arange(args.num_steps))[
    #     : args.num_minibatches * args.minibatch_size
    # ].reshape((args.num_minibatches, args.minibatch_size))

    rollout = jax.tree.map(partial(jnp.take, indices=indices, axis=0), rollout)

    return rollout


def train_on_rollout(
    agent: CDEAgent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: RolloutBuffer,
    args: PPOArguments,
    key: Key,
) -> tuple[CDEAgent, optax.OptState, PPOStats]:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    def scan_epoch(
        carry: tuple[PyTree, optax.OptState, Key],
        _: None,
    ) -> tuple[tuple[PyTree, optax.OptState, Key], PPOStats]:
        agent_dynamic, opt_state, key = carry
        agent = eqx.combine(agent_dynamic, agent_static)
        carry_key, batch_key = jr.split(key, 2)

        agent, opt_state, stats = epoch(
            agent,
            optimizer,
            opt_state,
            rollout,
            args,
            batch_key,
        )

        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)
        return (agent_dynamic, opt_state, carry_key), stats

    (agent_dynamic, opt_state, _), stats = jax.lax.scan(
        scan_epoch, (agent_dynamic, opt_state, key), length=args.num_epochs
    )

    stats = jax.tree.map(jnp.mean, stats)
    agent = eqx.combine(agent_dynamic, agent_static)

    stats.variance = jnp.var(rollout.returns)
    stats.explained_variance = jnp.where(
        stats.variance == 0,
        0.0,
        1 - jnp.var(rollout.returns - rollout.values) / stats.variance,
    )

    return agent, opt_state, stats


def add_scalar(
    writer: SummaryWriter | None,
    name: str,
    value: Float[Array, ""],
    global_step: Int[Array, ""],
):
    if writer is not None:
        jax.debug.callback(
            lambda name, value, global_step: writer.add_scalar(
                name, float(value), int(global_step)
            ),
            name,
            value,
            global_step,
        )


def log_callback(logger, name: str, value: Array):
    jax.debug.callback(lambda name, value: logger(f"{name}: {value}"), name, value)


def write_training_stats(
    writer: SummaryWriter | None,
    stats: PPOStats,
    global_step: Array,
    iteration: Int[Array, ""],
):
    log_callback(logger.info, "Completed iteration", iteration)
    log_callback(logger.debug, "global_step", global_step)

    log_callback(logger.debug, "total_loss", stats.total_loss)
    add_scalar(writer, "loss/total", stats.total_loss, global_step)
    log_callback(logger.debug, "policy_loss", stats.policy_loss)
    add_scalar(writer, "loss/policy", stats.policy_loss, global_step)
    log_callback(logger.debug, "value_loss", stats.value_loss)
    add_scalar(writer, "loss/value", stats.value_loss, global_step)
    log_callback(logger.debug, "entropy_loss", stats.entropy_loss)
    add_scalar(writer, "loss/entropy", stats.entropy_loss, global_step)
    log_callback(logger.debug, "state_loss", stats.state_loss)
    add_scalar(writer, "loss/state", stats.state_loss, global_step)
    log_callback(logger.debug, "learning_rate", stats.learning_rate)
    add_scalar(writer, "loss/learning_rate", stats.learning_rate, global_step)
    log_callback(logger.debug, "approx_kl", stats.approx_kl)
    add_scalar(writer, "stats/approx_kl", stats.approx_kl, global_step)
    log_callback(logger.debug, "variance", stats.variance)
    add_scalar(writer, "stats/variance", stats.variance, global_step)
    log_callback(logger.debug, "explained_variance", stats.explained_variance)
    add_scalar(
        writer, "stats/explained_variance", stats.explained_variance, global_step
    )
    log_callback(logger.debug, "grad_norm", stats.grad_norm)
    add_scalar(writer, "norm/grad", stats.grad_norm, global_step)
    log_callback(logger.debug, "update_norm", stats.update_norm)
    add_scalar(writer, "norm/update", stats.update_norm, global_step)


def train(
    env: environment.Environment | GymnaxWrapper,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    args: PPOArguments,
    key: Key,
) -> CDEAgent:
    logger.info(f"Training {args.run_name} with {args.num_iterations} iterations")
    env = wrappers.LogWrapper(env)
    writer = SummaryWriter(f"runs/{args.run_name}") if args.tb_logging else None
    if writer is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    if args.anneal_learning_rate:
        schedule = optax.schedules.cosine_decay_schedule(
            args.learning_rate, args.num_iterations * args.num_epochs * args.num_batches
        )
    else:
        schedule = optax.constant_schedule(args.learning_rate)

    adam = optax.inject_hyperparams(optax.adam)(learning_rate=schedule, eps=1e-5)
    optimizer = optax.named_chain(
        ("clipping", optax.clip_by_global_norm(args.max_gradient_norm)), ("adam", adam)
    )

    def scan_step(
        carry: tuple[TrainingState, EpisodeState, PyTree, Key],
        xs: Int[Array, ""],
    ) -> tuple[tuple[TrainingState, EpisodeState, PyTree, Key], None]:
        training_state, episode_state, agent_dynamic, key = carry
        agent = eqx.combine(agent_dynamic, agent_static)
        carry_key, rollout_key, training_key = jr.split(key, 3)

        episode_state, training_state, rollout = collect_rollout(
            env,
            env_params,
            agent,
            episode_state,
            training_state,
            args,
            key=rollout_key,
            writer=writer,
        )

        rollout.advantages, rollout.returns = calculate_gae(
            rollout.rewards, rollout.values, rollout.terminations, args
        )

        agent, training_state.opt_state, stats = train_on_rollout(
            agent, optimizer, training_state.opt_state, rollout, args, key=training_key
        )

        write_training_stats(writer, stats, training_state.global_step, xs)

        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)

        return (
            training_state,
            episode_state,
            agent_dynamic,
            carry_key,
        ), None

    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))
    reset_key, training_key = jr.split(key, 2)
    training_state = make_empty(
        TrainingState, global_step=jnp.array(0), opt_state=opt_state
    )
    episode_state = reset_episode(env, env_params, args, key=reset_key)

    (training_state, episode_state, agent_dynamic, _), _ = jax.lax.scan(
        scan_step,
        (training_state, episode_state, agent_dynamic, training_key),
        jnp.arange(args.num_iterations),
    )

    agent = eqx.combine(agent_dynamic, agent_static)

    return agent


class TransformObservationWrapper(GymnaxWrapper):
    def __init__(
        self,
        env: environment.Environment | GymnaxWrapper,
        func: Callable[[Any, Any, Array | None], Array],
        observation_space: spaces.Box | None = None,
    ):
        super().__init__(env)
        self.func = func
        self._observation_space = observation_space

    def observation_space(self, params) -> spaces.Box:
        if self._observation_space is not None:
            return self._observation_space
        else:
            return self._env.observation_space(params)

    def reset(self, key: Array, params: environment.EnvParams) -> tuple[Array, Any]:
        env_key, wrapper_key = jr.split(key)
        obs, state = self._env.reset(env_key, params)
        return self.func(obs, params, wrapper_key), state

    def step(
        self, key: Array, state: Any, action: Array, params: environment.EnvParams
    ) -> tuple[Array, Any, Array, Array, dict[Any, Any]]:
        env_key, wrapper_key = jr.split(key)
        obs, state, reward, done, info = self._env.step(env_key, state, action, params)
        return self.func(obs, params, wrapper_key), state, reward, done, info


class TransformRewardWrapper(GymnaxWrapper):
    def __init__(
        self,
        env: environment.Environment | GymnaxWrapper,
        func: Callable[[Any, Any, Array | None], Array],
    ):
        super().__init__(env)
        self.func = func

    def step(
        self, key: Array, state: Any, action: Array, params: environment.EnvParams
    ) -> tuple[Array, Any, Array, Array, dict[Any, Any]]:
        env_key, wrapper_key = jr.split(key)
        obs, state, reward, done, info = self._env.step(env_key, state, action, params)
        reward = self.func(reward, params, wrapper_key)
        return obs, state, reward, done, info


def run(
    env: environment.Environment | GymnaxWrapper,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    args: PPOArguments,
    *,
    key: Key,
) -> tuple[EpisodeState, RolloutBuffer]:
    episode_state = reset_episode(env, env_params, args, key=key)

    def scan_step(
        carry: tuple[EpisodeState, Key],
        _,
    ) -> tuple[tuple[EpisodeState, Key], tuple[RolloutBuffer, EpisodeState]]:
        episode_state, key = carry
        carry_key, step_key = jr.split(key, 2)

        episode_state, buffer, _ = env_step(
            env, env_params, agent, episode_state, args, key=step_key
        )

        return (episode_state, carry_key), (buffer, episode_state)

    (episode_state, _), (rollout, episode_states) = jax.lax.scan(
        scan_step, (episode_state, key), length=args.num_steps
    )
    return episode_states, rollout


def evaluate_episodes(
    env: environment.Environment | GymnaxWrapper,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    args: PPOArguments,
    num_episodes: int,
    *,
    key: Key,
) -> Float[Array, ""]:
    def episode_reward(key: Key):
        _, rollout = run(env, env_params, agent, args, key=key)
        episode_end = jnp.argmax(rollout.terminations | rollout.truncations)
        episode_end = jnp.where(episode_end == 0, args.num_steps, episode_end)
        mask = jnp.where(jnp.arange(args.num_steps) < episode_end, 1.0, 0.0)
        return jnp.mean(rollout.rewards * mask)

    return jnp.mean(jax.vmap(episode_reward)(jr.split(key, num_episodes)))


def visualize_episode(
    agent: CDEAgent,
    num_steps: int,
    *,
    key: Key,
):
    import gymnasium as gym
    import numpy as np

    env = gym.make("Pendulum-v1", render_mode="human")
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    observation, _ = env.reset()

    observations = jnp.full((num_steps,) + env.observation_space.shape, jnp.nan)
    observations = observations.at[0].set(jnp.asarray(observation))

    times = jnp.full((num_steps,), jnp.nan)
    times = times.at[0].set(0.0)

    for step in range(num_steps):
        _, action, _, _, _ = agent.get_action_and_value(
            ts=times,
            xs=observations,
        )
        clipped_action = jnp.clip(action, env.action_space.low, env.action_space.high)
        observation, _, termination, truncation, _ = env.step(np.array(clipped_action))

        observations = observations.at[step + 1].set(jnp.asarray(observation))
        times = times.at[step + 1].set(times[step] + args.agent_timestep)

        if termination or truncation:
            observation, _ = env.reset()

            observations = jnp.full((num_steps,) + env.observation_space.shape, jnp.nan)
            observations = observations.at[0].set(jnp.asarray(observation))

            times = jnp.full((args.num_steps,), jnp.nan)
            times = times.at[0].set(0.0)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    key = jr.key(0)
    env, env_params = gym.make("Pendulum-v1")

    agent = CDEAgent(
        env=env,
        env_params=env_params,
        hidden_size=4,
        width_size=64,
        depth=2,
        key=key,
    )

    args = PPOArguments(run_name="cde-agent", num_iterations=1024)

    agent = train(env, env_params, agent, args, key)
    visualize_episode(agent, 2000, key=key)
