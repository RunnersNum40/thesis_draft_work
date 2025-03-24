import dataclasses
import logging
from abc import abstractmethod

import chex
import equinox as eqx
import gymnax as gym
import jax
import optax
from gymnax.environments import environment
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key, PyTree

from neural_cde import NeuralCDE

logger = logging.getLogger(__name__)


class AbstractActorCriticAgent(eqx.Module, strict=True):
    """Abstract class for an actor-critic agent.

    Defines the basic interface for an actor-critic agent.
    """

    input_size: eqx.AbstractVar[int]
    action_size: eqx.AbstractVar[int]


class AbstractStatefulActorCriticAgent(AbstractActorCriticAgent, strict=True):
    state_size: eqx.AbstractVar[int]

    @abstractmethod
    def get_value(
        self,
        ts: Float[ArrayLike, " N"],
        xs: Float[ArrayLike, " N {self.input_size}"],
        z0: Float[Array, " {self.state_size}"],
        *args,
        key: Key | None = None,
        **kwargs,
    ) -> Float[Array, ""]:
        """Return the value of the given inputs."""
        raise NotImplementedError

    @abstractmethod
    def get_action_and_value(
        self,
        ts: Float[ArrayLike, " N"],
        xs: Float[ArrayLike, " N {self.input_size}"],
        z0: Float[Array, " {self.state_size}"],
        a1: Float[ArrayLike, " {self.action_size}"] | None = None,
        *args,
        key: Key | None = None,
        **kwargs,
    ) -> tuple[
        Float[Array, " {self.state_size}"],
        Float[Array, " {self.action_size}"],
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ]:
        """Return a final action and value for the given inputs.

        If an action is provided, it assumed to be fixed and the
        value is computed for the given action. Otherwise, the action is
        computed from the given inputs and sampled from the action distribution
        of the actor-critic model over the inputs.

        Arguments:
        - ts: Time steps of the inputs.
        - xs: Inputs to the actor-critic model.
        - z0: Initial state of the actor-critic model.
        - a1: Optional fixed action.
        - Key: Key for sampling the action.

        Returns:
        - z1: Final state of the actor-critic model.
        - a1: Action computed from the inputs.
        - log prob: Log probability of the action.
        - entropy: Entropy of the action distribution.
        - value: Value of the final state.
        """
        raise NotImplementedError

    @abstractmethod
    def initial_state(
        self,
        t0: Float,
        x0: Float[ArrayLike, " {self.input_size}"],
        *,
        key: Key | None = None,
    ) -> Float[Array, " {self.state_size}"]:
        """Generate an initial state z_0 from an inital input x_0.

        Arguments:
        - x0: Initial input to the actor-critic model.
        - key: Optional random key.

        Returns:
        - z0: Initial state of the actor-critic model.
        """
        raise NotImplementedError


class CDEAgent(AbstractStatefulActorCriticAgent, strict=True):
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
        env_params: gym.EnvParams,
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
            width_size=actor_width_size or width_size,
            depth=actor_depth or depth,
            key=actor_key,
        )

        self.action_std = jnp.zeros(self.action_size)

        self.critic = eqx.nn.MLP(
            in_size=processed_size,
            out_size="scalar",
            width_size=critic_width_size or width_size,
            depth=critic_depth or depth,
            key=critic_key,
        )

    def get_value(
        self,
        ts: Float[ArrayLike, " N"],
        xs: Float[ArrayLike, " N {self.input_size}"],
        z0: Float[Array, " {self.state_size}"],
        *,
        key: Key | None = None,
    ) -> Float[Array, ""]:
        """Return the value of the given inputs.

        Arguments:
        - ts: Time steps of the inputs.
        - xs: Inputs to the actor-critic model.
        - z0: Initial state of the actor-critic model.
        - key: Key for sampling the action.

        Returns:
        - value: Value of the final state.
        """
        _, processed = self.neural_cde(jnp.asarray(ts), jnp.asarray(xs), z0)
        value = self.critic(processed)

        return value

    def get_action_and_value(
        self,
        ts: Float[ArrayLike, " N"],
        xs: Float[ArrayLike, " N {self.input_size}"],
        z0: Float[Array, " {self.state_size}"],
        a1: Float[ArrayLike, " {self.action_size}"] | None = None,
        *,
        key: Key | None = None,
    ) -> tuple[
        Float[Array, " {self.state_size}"],
        Float[Array, " {self.action_size}"],
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ]:
        """Return a final action and value for the given inputs.

        If an action is provided, it assumed to be fixed and the
        value is computed for the given action. Otherwise, the action is
        computed from the given inputs and sampled from the action distribution
        of the actor-critic model over the inputs.

        Arguments:
        - ts: Time steps of the inputs.
        - xs: Inputs to the actor-critic model.
        - z0: Initial state of the actor-critic model.
        - a1: Optional fixed action.
        - key: Key for sampling the action.

        Returns:
        - z1: Final state of the actor-critic model.
        - a1: Action computed from the inputs.
        - log prob: Log probability of the action.
        - entropy: Entropy of the action distribution.
        - value: Value of the final state.
        """
        z1, processed = self.neural_cde(jnp.asarray(ts), jnp.asarray(xs), z0)
        action_mean = self.actor(processed)
        action_std = jnp.exp(self.action_std)
        value = self.critic(processed)

        if a1 is None:
            if key is not None:
                a1 = jr.normal(key, action_mean.shape) * action_std + action_mean
            else:
                a1 = action_mean
        else:
            a1 = jnp.asarray(a1)

        log_prob = jnp.sum(jax.scipy.stats.norm.logpdf(a1, action_mean, action_std))
        entropy = jnp.sum(0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1))

        return z1, a1, log_prob, entropy, value

    def initial_state(
        self,
        t0: Float[ArrayLike, ""],
        x0: Float[ArrayLike, " {self.input_size}"],
        *,
        key: Key | None = None,
    ) -> Float[Array, " {self.state_size}"]:
        """Generate an initial state z_0 from an inital input x_0.

        Arguments:
        - x0: Initial input to the actor-critic model.
        - key: Optional random key.

        Returns:
        - z0: Initial state of the actor-critic model.
        """
        return self.neural_cde.initial_state(jnp.asarray(t0), jnp.asarray(x0))


@chex.dataclass
class RolloutBuffer:
    """Meant to be used as a return value for a rollout scan."""

    observations: Float[Array, " *N observation_size"]
    actions: Float[Array, " *N action_size"]
    log_probs: Float[Array, " *N"]
    entropies: Float[Array, " *N"]
    values: Float[Array, " *N"]
    rewards: Float[Array, " *N"]
    terminations: Bool[Array, " *N"]
    truncations: Bool[Array, " *N"]
    advantages: Float[Array, " *N"]
    returns: Float[Array, " *N"]


@chex.dataclass
class StatefulRolloutBuffer(RolloutBuffer):
    times: Float[Array, " *N"]
    states: Float[Array, " *N state_size"]


@chex.dataclass
class TrainingState:
    """Meant to be used as a carry value during training or rollout scanning."""

    env_state: gym.EnvState
    opt_state: optax.OptState
    observation: Float[Array, " observation_size"]
    global_step: Int[Array, ""]


@chex.dataclass
class StatefulAgentTrainingState(TrainingState):
    last_observation: Float[Array, " observation_size"]
    agent_time: Float[Array, ""]
    last_agent_time: Float[Array, ""]
    agent_state: Float[Array, " state_size"]


@chex.dataclass
class PPOStats:
    total_loss: Float[Array, " *N"]
    policy_loss: Float[Array, " *N"]
    value_loss: Float[Array, " *N"]
    entropy_loss: Float[Array, " *N"]
    approx_kl: Float[Array, " *N"]


@chex.dataclass
class PPOArguments:
    num_steps: int
    gamma: float
    gae_lambda: float
    num_epochs: int
    normalize_advantage: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    value_coefficient: float
    max_gradient_norm: float
    target_kl: float | None
    minibatch_size: int
    num_iterations: int


def make_empty(cls, **kwargs):
    """Create an empty dataclass with the fields set to None.

    If a field is provided in kwargs, it will be set to the value in kwargs.
    """
    dcls = cls(**{f.name: None for f in dataclasses.fields(cls)})
    for k, v in kwargs.items():
        dcls.__setattr__(k, v)

    return dcls


def reset(
    env: environment.Environment,
    env_params: gym.EnvParams,
    agent: AbstractStatefulActorCriticAgent,
    training_state: StatefulAgentTrainingState,
    *,
    key: Key,
) -> StatefulAgentTrainingState:
    """Reset the environment and agent to a fresh state.

    Arguments:
    - env: Environment to reset.
    - env_params: Parameters for the environment.
    - agent: Agent to reset.
    - training_state: Current training state.
    - key: Random key for resetting the environment.

    Returns:
    - training_state: A fresh training state.
    """
    env_key, agent_key = jr.split(key, 2)
    observation, env_state = env.reset(env_key, env_params)

    agent_time = jnp.array(0.0)
    agent_state = agent.initial_state(agent_time, observation, key=agent_key)

    return StatefulAgentTrainingState(
        env_state=env_state,
        opt_state=training_state.opt_state,
        observation=observation,
        last_observation=observation,
        agent_time=agent_time,
        last_agent_time=agent_time - 1e-3,
        agent_state=agent_state,
        global_step=training_state.global_step,
    )


def env_step(
    env: environment.Environment,
    env_params: gym.EnvParams,
    agent: AbstractStatefulActorCriticAgent,
    training_state: StatefulAgentTrainingState,
    *,
    key: Key,
) -> tuple[StatefulAgentTrainingState, StatefulRolloutBuffer]:
    """Step the environment and agent forward one step and store the data in a rollout buffer.

    Arguments:
    - env: Environment to interact with.
    - env_params: Parameters for the environment.
    - agent: Agent to interact with the environment.
    - training_state: Current training state.
    - key: Random key for sampling.

    Returns:
    - [carry[training_state, key], rollout_step]: New training state and key, and rollout buffer with the current step data.
    """
    actor_key, env_key, reset_key = jr.split(key, 3)

    # Store the current state
    buffer = make_empty(StatefulRolloutBuffer)
    buffer.observations = training_state.observation
    buffer.times = training_state.agent_time
    buffer.states = training_state.agent_state

    # Get the agent action and state from the current state
    agent_state, action, log_prob, entropy, value = agent.get_action_and_value(
        jnp.asarray([training_state.last_agent_time, training_state.agent_time]),
        jnp.asarray([training_state.last_observation, training_state.observation]),
        training_state.agent_state,
        key=actor_key,
    )
    # Store the agent action and state
    buffer.actions = action
    buffer.log_probs = log_prob
    buffer.entropies = entropy
    buffer.values = value

    # Get the next state and reward from the environment
    clipped_action = jnp.clip(
        action, env.action_space(env_params).low, env.action_space(env_params).high
    )
    observation, env_state, reward, done, info = env.step(
        env_key, training_state.env_state, clipped_action, env_params
    )
    termination = done
    truncation = jnp.array(False)
    # Store the environment state and reward
    buffer.rewards = reward
    buffer.terminations = termination
    buffer.truncations = truncation

    # Update the training state
    training_state.env_state = env_state
    training_state.last_observation = training_state.observation
    training_state.observation = observation
    training_state.last_agent_time = training_state.agent_time
    training_state.agent_time += 1.0
    training_state.agent_state = agent_state
    training_state.global_step += 1

    # Reset if needed
    training_state = jax.lax.cond(
        jnp.logical_or(termination, truncation),
        lambda: reset(env, env_params, agent, training_state, key=reset_key),
        lambda: training_state,
    )

    return training_state, buffer


def collect_stateful_ppo_rollout(
    env: environment.Environment,
    env_params: gym.EnvParams,
    agent: AbstractStatefulActorCriticAgent,
    training_state: StatefulAgentTrainingState,
    args: PPOArguments,
    *,
    key: Key,
) -> tuple[StatefulAgentTrainingState, StatefulRolloutBuffer]:
    def scan_env_step(
        carry: tuple[StatefulAgentTrainingState, Key], _
    ) -> tuple[tuple[StatefulAgentTrainingState, Key], StatefulRolloutBuffer]:
        training_state, key = carry
        carry_key, step_key = jr.split(key, 2)

        training_state, rollout_step = env_step(
            env, env_params, agent, training_state, key=step_key
        )

        return (training_state, carry_key), rollout_step

    (training_state, _), rollout = jax.lax.scan(
        scan_env_step,
        (training_state, key),
        length=args.num_steps,
    )

    return training_state, rollout


def compute_gae(
    agent: AbstractStatefulActorCriticAgent,
    rollout: StatefulRolloutBuffer,
    training_state: StatefulAgentTrainingState,
    args: PPOArguments,
) -> StatefulRolloutBuffer:
    next_value = agent.get_value(
        ts=jnp.asarray([training_state.last_agent_time, training_state.agent_time]),
        xs=jnp.asarray([training_state.last_observation, training_state.observation]),
        z0=training_state.agent_state,
    )

    next_non_terminals = jnp.concatenate(
        [
            1.0
            - jnp.logical_or(rollout.terminations[1:], rollout.truncations[1:]).astype(
                jnp.float32
            ),
            jnp.array(
                [
                    1.0
                    - jnp.logical_or(
                        rollout.terminations[-1], rollout.truncations[-1]
                    ).astype(jnp.float32)
                ]
            ),
        ],
        axis=0,
    )

    next_values = jnp.concatenate([rollout.values[1:], jnp.array([next_value])], axis=0)

    def scan_fn(
        carry: Float[Array, ""],
        x: tuple[
            Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]
        ],
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        reward, value, next_non_terminal, next_value_ = x
        delta = reward + args.gamma * next_value_ - value
        new_carry = delta + args.gamma * args.gae_lambda * next_non_terminal * carry
        return new_carry, new_carry

    _, advantages_reversed = jax.lax.scan(
        scan_fn,
        jnp.array(0.0),
        (rollout.rewards, rollout.values, next_non_terminals, next_values),
        reverse=True,
    )
    advantages = jnp.flip(advantages_reversed, axis=0)
    returns = advantages + rollout.values

    rollout = dataclasses.replace(rollout, advantages=advantages, returns=returns)
    return rollout


def loss(
    agent: AbstractStatefulActorCriticAgent,
    training_state: StatefulAgentTrainingState,
    rollout: StatefulRolloutBuffer,
    args: PPOArguments,
) -> tuple[Float[Array, ""], PPOStats]:
    times_full = jnp.concatenate(
        [rollout.times, jnp.expand_dims(training_state.agent_time, axis=0)]
    )
    obs_full = jnp.concatenate(
        [rollout.observations, jnp.expand_dims(training_state.observation, axis=0)]
    )

    t_pairs = jnp.stack([times_full[:-1], times_full[1:]], axis=1)
    x_pairs = jnp.stack([obs_full[:-1], obs_full[1:]], axis=1)

    _, _, new_log_probs, new_entropies, new_values = jax.vmap(
        agent.get_action_and_value
    )(t_pairs, x_pairs, rollout.states, rollout.actions)

    log_ratio = new_log_probs - rollout.log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = jnp.mean(ratio - log_ratio) - 1.0

    if args.normalize_advantage:
        advantages = (rollout.advantages - jnp.mean(rollout.advantages)) / (
            jnp.std(rollout.advantages) + 1e-8
        )
    else:
        advantages = rollout.advantages

    policy_loss = -jnp.mean(
        jnp.minimum(
            advantages * ratio,
            advantages
            * jnp.clip(ratio, 1.0 - args.clip_coefficient, 1.0 + args.clip_coefficient),
        )
    )

    if args.clip_value_loss:
        clipped_values = rollout.values + jnp.clip(
            new_values - rollout.values, -args.clip_coefficient, args.clip_coefficient
        )
        value_loss = (
            jnp.mean(
                jnp.maximum(
                    jnp.square(new_values - rollout.returns),
                    jnp.square(clipped_values - rollout.returns),
                )
            )
            / 2.0
        )
    else:
        value_loss = jnp.mean(jnp.square(new_values - rollout.returns)) / 2.0

    entropy_loss = jnp.mean(new_entropies)
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


loss_grad = eqx.filter_value_and_grad(loss, has_aux=True)


def get_batch_indices(
    batch_size: int, dataset_size: int, *, key: Key | None = None
) -> Int[Array, " {dataset_size // batch_size} {batch_size}"]:
    indices = jnp.arange(dataset_size)

    if key is not None:
        indices = jax.random.permutation(key, indices)

    if dataset_size % batch_size != 0:
        logger.warning("Dataset size is not divisible by batch size.")

    indices = indices[: dataset_size - (dataset_size % batch_size)]
    return indices.reshape(-1, batch_size)


def train_on_batch(
    agent: AbstractStatefulActorCriticAgent,
    training_state: StatefulAgentTrainingState,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: StatefulRolloutBuffer,
    batch_indices: Int[Array, " N"],
    args: PPOArguments,
) -> tuple[AbstractStatefulActorCriticAgent, optax.OptState, PPOStats]:
    rollout = jax.tree.map(lambda x: x[batch_indices], rollout)
    (_, stats), grads = loss_grad(agent, training_state, rollout, args)
    updates, opt_state = optimizer.update(grads, opt_state)
    agent = eqx.apply_updates(agent, updates)

    return agent, opt_state, stats


def epoch(
    agent: AbstractStatefulActorCriticAgent,
    training_state: StatefulAgentTrainingState,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: StatefulRolloutBuffer,
    args: PPOArguments,
    *,
    key: Key | None = None,
) -> tuple[AbstractStatefulActorCriticAgent, optax.OptState, PPOStats]:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    def scan_batch(
        carry: tuple[PyTree, optax.OptState], xs: Array
    ) -> tuple[tuple[PyTree, optax.OptState], PPOStats]:
        agent_dynamic, opt_state = carry
        agent = eqx.combine(agent_dynamic, agent_static)
        agent, opt_state, stats = train_on_batch(
            agent, training_state, optimizer, opt_state, rollout, xs, args
        )
        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)
        return (agent_dynamic, opt_state), stats

    indices = get_batch_indices(
        args.minibatch_size, rollout.observations.shape[0], key=key
    )
    (agent_dynamic, opt_state), stats = jax.lax.scan(
        scan_batch, (agent_dynamic, opt_state), indices
    )

    stats = jax.tree_map(jnp.mean, stats)
    agent = eqx.combine(agent_dynamic, agent_static)

    return agent, opt_state, stats


def train_on_rollout(
    agent: AbstractStatefulActorCriticAgent,
    training_state: StatefulAgentTrainingState,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: StatefulRolloutBuffer,
    args: PPOArguments,
    *,
    key: Key | None = None,
) -> tuple[AbstractStatefulActorCriticAgent, optax.OptState, PPOStats]:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    def scan_epoch(
        carry: tuple[PyTree, optax.OptState, Key | None],
        xs: None,
    ) -> tuple[tuple[PyTree, optax.OptState, Key | None], PPOStats]:
        agent_dynamic, opt_state, key = carry
        agent = eqx.combine(agent_dynamic, agent_static)

        if key is not None:
            carry_key, batch_key = jr.split(key, 2)
        else:
            carry_key, batch_key = None, None

        agent, opt_state, stats = epoch(
            agent, training_state, optimizer, opt_state, rollout, args, key=batch_key
        )

        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)

        return (agent_dynamic, opt_state, carry_key), stats

    (agent_dynamic, opt_state, _), stats = jax.lax.scan(
        scan_epoch, (agent_dynamic, opt_state, key), length=args.num_epochs
    )

    stats = jax.tree_map(jnp.mean, stats)
    agent = eqx.combine(agent_dynamic, agent_static)

    return agent, opt_state, stats


def train(
    env: environment.Environment,
    env_params: gym.EnvParams,
    optimizer: optax.GradientTransformation,
    agent: AbstractStatefulActorCriticAgent,
    args: PPOArguments,
    *,
    key=Key,
) -> AbstractStatefulActorCriticAgent:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    def scan_step(
        carry: tuple[StatefulAgentTrainingState, PyTree, optax.OptState, Key], xs: None
    ) -> tuple[tuple[StatefulAgentTrainingState, PyTree, optax.OptState, Key], PyTree]:
        training_state, agent_dynamic, opt_state, key = carry
        agent = eqx.combine(agent_dynamic, agent_static)
        rollout_key, training_key = jr.split(key, 2)

        training_state, rollout = collect_stateful_ppo_rollout(
            env, env_params, agent, training_state, args, key=rollout_key
        )

        rollout = compute_gae(agent, rollout, training_state, args)

        agent, opt_state, stats = train_on_rollout(
            agent, training_state, optimizer, opt_state, rollout, args, key=training_key
        )

        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)

        return (training_state, agent_dynamic, opt_state, key), stats

    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))
    reset_key, training_key = jr.split(key, 2)
    training_state = make_empty(
        StatefulAgentTrainingState, global_step=jnp.array(0), opt_state=opt_state
    )
    training_state = reset(env, env_params, agent, training_state, key=reset_key)

    (training_state, agent_dynamic, opt_state, _), _ = jax.lax.scan(
        scan_step,
        (training_state, agent_dynamic, opt_state, training_key),
        length=args.num_iterations,
    )

    agent = eqx.combine(agent_dynamic, agent_static)

    return agent


if __name__ == "__main__":
    args = make_empty(
        PPOArguments,
        num_steps=1024,
        gamma=0.93,
        gae_lambda=0.95,
        num_epochs=4,
        normalize_advantage=True,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        max_gradient_norm=0.5,
        target_kl=None,
        minibatch_size=64,
        num_iterations=1024,
    )

    key = jr.key(0)
    agent_key, reset_key, rollout_key = jr.split(key, 3)

    env, env_params = gym.make("Pendulum-v1")

    agent = CDEAgent(
        env=env,
        env_params=env_params,
        hidden_size=16,
        processed_size=16,
        width_size=64,
        depth=2,
        key=agent_key,
    )

    optimizer = optax.adam(1e-3)

    agent = train(env, env_params, optimizer, agent, args, key=rollout_key)
