"""ODE RNN PPO

An implementation of Proximal Policy Optimization with a Neural Controlled Differential Equations.
"""

import dataclasses
import logging
from functools import partial
from typing import Literal

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
        weight_scale: float = 1.0,
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

        self.tensor_mlp = eqx.tree_at(
            lambda tree: [linear.weight for linear in tree.mlp.layers],
            self.tensor_mlp,
            replace_fn=lambda x: x * weight_scale,
        )

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

        # Default to a linear initial condition
        self.initial = eqx.nn.MLP(
            in_size=input_size + 1,
            out_size=hidden_size,
            width_size=initial_width_size or width_size,
            depth=initial_depth if initial_depth is not None else 0,
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
            width_size=(
                output_width_size if output_width_size is not None else width_size
            ),
            depth=output_depth if output_depth is not None else depth,
            key=okey,
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
        if evolving_out:
            ts = eqx.error_if(ts, jnp.isnan(ts).any(), "ts contains NaNs")
            xs = eqx.error_if(xs, jnp.isnan(xs).any(), "xs contains NaNs")

        if self.input_size == "scalar":
            assert xs.ndim == 1
            xs = jnp.expand_dims(xs, axis=-1)

        assert ts.ndim == 1
        assert xs.ndim == 2
        assert z0.ndim == 1
        assert xs.shape[0] == ts.shape[0]

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
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

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
            dt0=None,
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
    neural_cde: NeuralCDE
    actor: eqx.nn.MLP
    action_std: Float[Array, " action_size"]
    critic: eqx.nn.MLP

    def __init__(
        self,
        env: environment.Environment,
        env_params: gym.EnvParams | wrappers.purerl.GymnaxWrapper,
        hidden_size: int,
        processed_size: int,
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
    ) -> None:
        """Create an actor critic model with a neural CDE.

        Uses the neural CDE to process inputs and generate a state
        that is shared between the actor and critic.

        Arguments:
        - env: Environment to interact with.
        - env_params: Parameters for the environment.
        - hidden_size: Size of the hidden state in the neural CDE.
        - processed_size: Size of the processed state shared between the actor and critic.
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
        """
        assert isinstance(env.action_space(env_params), spaces.Box)
        assert isinstance(env.observation_space(env_params), spaces.Box)

        cde_key, actor_key, critic_key = jr.split(key, 3)

        self.input_size = int(
            jnp.asarray(env.observation_space(env_params).shape).prod()
        )
        self.state_size = hidden_size
        self.action_size = int(jnp.asarray(env.action_space(env_params).shape).prod())

        self.neural_cde = NeuralCDE(
            input_size=self.input_size,
            hidden_size=hidden_size,
            output_size=processed_size,
            width_size=width_size,
            depth=depth,
            initial_width_size=initial_width_size,
            initial_depth=initial_depth,
            output_width_size=output_width_size,
            output_depth=output_depth,
            key=cde_key,
        )

        self.actor = eqx.nn.MLP(
            in_size=processed_size,
            out_size=self.action_size,
            width_size=actor_width_size if actor_width_size is not None else width_size,
            depth=actor_depth if actor_depth is not None else depth,
            key=actor_key,
        )

        self.action_std = jnp.zeros(self.action_size)

        self.critic = eqx.nn.MLP(
            in_size=processed_size,
            out_size="scalar",
            width_size=(
                critic_width_size if critic_width_size is not None else width_size
            ),
            depth=critic_depth if critic_depth is not None else depth,
            key=critic_key,
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
        z0 = z0 or self.neural_cde.initial_state(ts[0], xs[0])
        t1 = t1 or jnp.max(ts)

        z1, processed = self.neural_cde(
            jnp.asarray(ts), jnp.asarray(xs), z0=z0, t1=t1, evolving_out=evolving_out
        )
        if evolving_out:
            action_mean = jax.vmap(self.actor)(processed)
            value = jax.vmap(self.critic)(processed)
        else:
            action_mean = self.actor(processed)
            value = self.critic(processed)

        action_std = jnp.exp(self.action_std)

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
        return self.neural_cde.initial_state(jnp.asarray(t0), jnp.asarray(x0))


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
    num_steps: int
    num_epochs: int
    num_minibatches: int
    minibatch_size: int
    num_batches: int
    batch_size: int

    agent_timestep: float = 1e-1

    gamma: float = 0.99
    gae_lambda: float = 0.95

    learning_rate: float = 1e-3
    anneal_learning_rate: bool = True

    normalize_advantage: bool = True
    clip_coefficient: float = 0.2
    clip_value_loss: bool = True
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.8
    max_gradient_norm: float = 0.5
    target_kl: float | None = None


def make_empty(cls, **kwargs):
    """Create an empty dataclass with the fields set to None.

    If a field is provided in kwargs, it will be set to the value in kwargs.
    """
    dcls = cls(**{f.name: None for f in dataclasses.fields(cls)})
    for k, v in kwargs.items():
        dcls.__setattr__(k, v)

    return dcls


def reset_episode(
    env: wrappers.LogWrapper | environment.Environment,
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
    env: environment.Environment | wrappers.LogWrapper,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    episode_state: EpisodeState,
    args: PPOArguments,
    *,
    key: Key,
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

    _, action, log_prob, entropy, value = agent.get_action_and_value(
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
    observation, env_state, reward, done, info = env.step(
        env_key, episode_state.env_state, clipped_action, env_params
    )
    observation = jnp.clip(observation, -10.0, 10.0)
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
    env: environment.Environment | wrappers.LogWrapper,
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
            env, env_params, agent, episode_state, args, key=step_key
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


def calculate_loss(
    agent: CDEAgent,
    times: Float[Array, " num_steps"],
    observations: Float[Array, " num_steps observation_size"],
    actions: Float[Array, " num_steps action_size"],
    log_probs: Float[Array, " num_steps"],
    values: Float[Array, " num_steps"],
    advantages: Float[Array, " num_steps"],
    returns: Float[Array, " num_steps"],
    args: PPOArguments,
) -> tuple[Float[Array, ""], PPOStats]:
    """Relies on no terminations or truncations in the rollout buffer."""
    _, _, new_log_probs, entropies, new_values = agent.get_action_and_value(
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
            ratio * advantages,
            jnp.clip(ratio, 1 - args.clip_coefficient, 1 + args.clip_coefficient)
            * advantages,
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
    loss = (
        policy_loss
        + args.value_coefficient * value_loss
        - args.entropy_coefficient * entropy_loss
    )
    stats = make_empty(
        PPOStats,
        total_loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        approx_kl=approx_kl,
    )

    return loss, stats


def calculate_batch_loss(
    agent: CDEAgent,
    times: Float[Array, " N num_steps"],
    observations: Float[Array, " N num_steps observation_size"],
    actions: Float[Array, " N num_steps action_size"],
    log_probs: Float[Array, " N num_steps"],
    values: Float[Array, " N num_steps"],
    advantages: Float[Array, " N num_steps"],
    returns: Float[Array, " N num_steps"],
    args: PPOArguments,
) -> tuple[Float[Array, ""], PPOStats]:
    losses, stats = jax.vmap(partial(calculate_loss, agent, args=args))(
        times, observations, actions, log_probs, values, advantages, returns
    )
    return jnp.mean(losses), jax.tree.map(jnp.mean, stats)


calculate_batch_grad = eqx.filter_value_and_grad(calculate_batch_loss, has_aux=True)


def train_on_batch(
    agent: CDEAgent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    times: Float[Array, " N num_steps"],
    observations: Float[Array, " N num_steps observation_size"],
    actions: Float[Array, " N num_steps action_size"],
    log_probs: Float[Array, " N num_steps"],
    values: Float[Array, " N num_steps"],
    advantages: Float[Array, " N num_steps"],
    returns: Float[Array, " N num_steps"],
    args: PPOArguments,
) -> tuple[CDEAgent, optax.OptState, PPOStats]:
    (_, stats), grads = calculate_batch_grad(
        agent,
        times,
        observations,
        actions,
        log_probs,
        values,
        advantages,
        returns,
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
    batch_size: int, dataset_size: int, num_batches: int, key: Key | None = None
) -> Int[Array, " {num_batches} {batch_size}"]:
    if key is None:
        indices = jnp.arange(dataset_size)
    else:
        indices = jr.permutation(key, dataset_size)

    total_required = batch_size * num_batches
    tiled_indices = jnp.tile(indices, (total_required // dataset_size + 1,))
    selected = tiled_indices[:total_required]
    return selected.reshape((num_batches, batch_size))


def epoch(
    agent: CDEAgent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    times: Float[Array, " N num_steps"],
    observations: Float[Array, " N num_steps observation_size"],
    actions: Float[Array, " N num_steps action_size"],
    log_probs: Float[Array, " N num_steps"],
    values: Float[Array, " N num_steps"],
    advantages: Float[Array, " N num_steps"],
    returns: Float[Array, " N num_steps"],
    args: PPOArguments,
    key: Key | None,
) -> tuple[CDEAgent, optax.OptState, PPOStats]:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    def scan_batch(
        carry: tuple[PyTree, optax.OptState], xs: Int[Array, " batch_size"]
    ) -> tuple[tuple[PyTree, optax.OptState], PPOStats]:
        agent_dynamic, opt_state = carry
        agent = eqx.combine(agent_dynamic, agent_static)

        (
            times_batch,
            observations_batch,
            actions_batch,
            log_probs_batch,
            values_batch,
            advantages_batch,
            returns_batch,
        ) = jax.tree.map(
            lambda x: x[xs],
            (times, observations, actions, log_probs, values, advantages, returns),
        )

        agent, opt_state, stats = train_on_batch(
            agent,
            optimizer,
            opt_state,
            times_batch,
            observations_batch,
            actions_batch,
            log_probs_batch,
            values_batch,
            advantages_batch,
            returns_batch,
            args,
        )
        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)
        return (agent_dynamic, opt_state), stats

    indices = get_batch_indices(args.batch_size, times.shape[0], args.num_batches, key)
    (agent_dynamic, opt_state), stats = jax.lax.scan(
        scan_batch, (agent_dynamic, opt_state), indices
    )

    stats = jax.tree.map(jnp.mean, stats)
    agent = eqx.combine(agent_dynamic, agent_static)

    return agent, opt_state, stats


def minibatch_rollout(rollout: RolloutBuffer, args: PPOArguments, key: Key) -> tuple[
    Float[Array, "num_minibatches minibatch_size"],
    Float[Array, "num_minibatches minibatch_size observation_size"],
    Float[Array, "num_minibatches minibatch_size action_size"],
    Float[Array, "num_minibatches minibatch_size"],
    Float[Array, "num_minibatches minibatch_size"],
    Float[Array, "num_minibatches minibatch_size"],
    Float[Array, "num_minibatches minibatch_size"],
]:
    """Extract minibatches of contiguous sequences from a rollout buffer for PPO training.

    Given a rollout buffer, this function identifies and extracts sequences of consecutive
    steps without episode terminations or truncations. The sequences are grouped into
    minibatches suitable for batch training. If insufficient valid sequences are found, existing
    sequences are duplicated to meet the required batch size.

    Arguments:
    - rollout: Rollout buffer containing episode data.
    - args: PPO training arguments specifying:
        - `num_minibatches`: Number of minibatches to generate.
        - `minibatch_size`: Number of steps per minibatch.
    - key: Key for shuffling and duplication.

    Returns:
    - times: `[num_minibatches, minibatch_size]`
    - observations: `[num_minibatches, minibatch_size, observation_size]`
    - actions: `[num_minibatches, minibatch_size, action_size]`
    - log_probs: `[num_minibatches, minibatch_size]`
    - values: `[num_minibatches, minibatch_size]`
    - advantages: `[num_minibatches, minibatch_size]`
    - returns: `[num_minibatches, minibatch_size]`

    Notes:
        Each minibatch contains sequences that are strictly contiguous within an episode,
        ensuring temporal consistency required by sequential models.
    """
    episode_boundaries = rollout.terminations | rollout.truncations

    def is_valid_start(idx):
        offset = jnp.arange(args.minibatch_size - 1)
        indices = idx + offset

        in_bounds = (idx + args.minibatch_size - 1) < args.num_steps
        no_boundary = ~jnp.any(jnp.where(in_bounds, episode_boundaries[indices], True))

        return in_bounds & no_boundary

    all_indices = jnp.arange(args.num_steps - args.minibatch_size + 1)
    valid_mask = jax.vmap(is_valid_start)(all_indices)
    valid_indices = jnp.nonzero(valid_mask, size=all_indices.shape[0], fill_value=-1)[0]
    num_valid = jnp.sum(valid_mask)

    valid_indices = eqx.error_if(
        valid_indices,
        num_valid == 0,
        "No valid batches found",
    )

    final_start_indices = jr.choice(
        key,
        valid_indices,
        shape=(args.num_minibatches,),
        replace=True,
        p=(valid_indices >= 0) / jnp.sum(valid_indices >= 0),
    )

    offsets = jnp.arange(args.minibatch_size)
    final_indices = final_start_indices[:, None] + offsets[None, :]

    times = rollout.times[final_indices]
    observations = rollout.observations[final_indices]
    actions = rollout.actions[final_indices]
    log_probs = rollout.log_probs[final_indices]
    values = rollout.values[final_indices]
    advantages = rollout.advantages[final_indices]
    returns = rollout.returns[final_indices]

    return times, observations, actions, log_probs, values, advantages, returns


def train_on_rollout(
    agent: CDEAgent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: RolloutBuffer,
    args: PPOArguments,
    key: Key,
) -> tuple[CDEAgent, optax.OptState, PPOStats]:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)
    batching_key, epoch_key = jr.split(key, 2)

    (
        times,
        observations,
        actions,
        log_probs,
        values,
        advantages,
        returns,
    ) = minibatch_rollout(rollout, args, batching_key)

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
            times,
            observations,
            actions,
            log_probs,
            values,
            advantages,
            returns,
            args,
            batch_key,
        )

        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)
        return (agent_dynamic, opt_state, carry_key), stats

    (agent_dynamic, opt_state, _), stats = jax.lax.scan(
        scan_epoch, (agent_dynamic, opt_state, epoch_key), length=args.num_epochs
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
    env: environment.Environment,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    args: PPOArguments,
    key: Key,
) -> CDEAgent:
    env = wrappers.LogWrapper(env)
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_hparams(vars(args), {})

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


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    key = jr.key(0)
    env, env_params = gym.make("Pendulum-v1")

    agent = CDEAgent(
        env=env,
        env_params=env_params,
        hidden_size=2,
        processed_size=2,
        width_size=16,
        depth=1,
        key=key,
        actor_width_size=8,
        actor_depth=0,
        output_width_size=8,
        output_depth=0,
        critic_width_size=8,
        critic_depth=1,
    )

    args = PPOArguments(
        run_name="cde",
        num_iterations=512,
        num_steps=1024,
        num_epochs=16,
        num_minibatches=32,
        minibatch_size=64,
        num_batches=8,
        batch_size=4,
    )

    agent = train(env, env_params, agent, args, key)
