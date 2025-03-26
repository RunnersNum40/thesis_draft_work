import dataclasses
import logging
from functools import partial

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

    def get_action_and_value(
        self,
        ts: Float[Array, " N"],
        xs: Float[Array, " N {self.input_size}"],
        a1: Float[Array, " {self.action_size}"] | None = None,
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

        A set of times and inputs padded with NaNs is provided to allow for
        JIT compilation since the number of steps is not known at compile time.
        Pass the maximum index of the inputs to act on as `max_index`.

        Arguments:
        - ts: Time steps of the inputs.
        - xs: Inputs to the actor-critic model.
        - a1: Optional fixed action.
        - key: Key for sampling the action.
        - evolving_out: Whether to compute the output for every time step.

        Returns:
        - z1: Final state of the actor-critic model.
        - a1: Action computed from the inputs.
        - log prob: Log probability of the action.
        - entropy: Entropy of the action distribution.
        - value: Value of the final state.
        """
        z0 = self.initial_state(ts[0], xs[0])
        z1, processed, _ = self.neural_cde(
            jnp.asarray(ts), jnp.asarray(xs), z0, evolving_out=evolving_out
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
class EpisodeState:
    step: Int[Array, ""]
    env_state: gym.EnvState
    observations: Float[Array, " num_steps observation_size"]
    times: Float[Array, " num_steps"]


@chex.dataclass
class EpisodesRollout:
    """Rollout of multiple episodes.

    The history of the environment and agent states are stored in the rollout
    and the initial state. This can be used to train the agent on rollouts with
    a full backpropagation through time.

    Episodes can be padded with NaNs to match the number of steps in the PPO arguments.
    Additionally extra episodes can be padded to match the number of steps in the PPO arguments.
    """

    observations: Float[Array, " *N num_steps observation_size"]
    actions: Float[Array, " *N num_steps action_size"]
    log_probs: Float[Array, " *N num_steps"]
    entropies: Float[Array, " *N num_steps"]
    values: Float[Array, " *N num_steps"]
    rewards: Float[Array, " *N num_steps"]
    terminations: Bool[Array, " *N num_steps"]
    truncations: Bool[Array, " *N num_steps"]
    advantages: Float[Array, " *N num_steps"]
    returns: Float[Array, " *N num_steps"]
    times: Float[Array, " *N num_steps"]


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
    learning_rate: float
    anneal_learning_rate: bool


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
    args: PPOArguments,
    *,
    key: Key,
) -> EpisodeState:
    """Reset the environment and agent to a fresh state.

    Initializes the time to a random value between 0 and 1.

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
    observations = jnp.full(
        (args.num_steps, observation.shape[0]), jnp.nan, dtype=observation.dtype
    )
    observations = observations.at[0].set(observation)

    agent_times = jnp.full((args.num_steps,), jnp.nan, dtype=jnp.array(0.0).dtype)
    agent_times = agent_times.at[0].set(
        jr.uniform(time_key, (), minval=0.0, maxval=1.0)
    )

    return EpisodeState(
        step=jnp.array(0),  # Start at 1 to account for the initial state.
        env_state=env_state,
        observations=observations,
        times=agent_times,
    )


def rollover_episode_state(episode_state: EpisodeState) -> EpisodeState:
    """If an episode has already filled a buffer, rollover the episode state without resetting the environment."""
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
    env: environment.Environment,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    episode_state: EpisodeState,
    args: PPOArguments,
    *,
    key: Key,
) -> tuple[EpisodeState, EpisodesRollout]:
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
    """
    actor_key, env_key, reset_key = jr.split(key, 3)

    _, action, log_prob, entropy, value = agent.get_action_and_value(
        episode_state.times,
        episode_state.observations,
        key=actor_key,
    )

    buffer = make_empty(
        EpisodesRollout,
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
            episode_state.times[episode_state.step] + 1e-1
        ),
    )

    # If the episode is done, reset the environment.
    episode_state = jax.lax.cond(
        termination | truncation,
        lambda: reset(env, env_params, args, key=reset_key),
        lambda: episode_state,
    )

    return episode_state, buffer


def split_into_episodes(
    rollout: EpisodesRollout, args: PPOArguments
) -> EpisodesRollout:
    """Split a rollout of multiple episodes into multiple episodes.

    Episodes are padded to `args.num_steps`.
    The number of episodes is also padded to `args.num_steps`.

    Arguments:
    - rollout: Rollout of multiple concatenated episodes.
    - args: PPO arguments.

    Returns:
    - rollout: Batched rollout of episodes.
      Padded to match the number of steps in the PPO arguments.
    """

    def body_fn(
        carry: tuple[int, int, PyTree], idx: Int[Array, ""]
    ) -> tuple[tuple[int, int, PyTree], None]:
        """Fill the batched rollout with the data from the current step."""
        ep_idx, step_idx, out_tree = carry

        current_data = jax.tree.map(lambda x: x[idx], rollout)
        out_tree = jax.tree.map(
            lambda arr, x: arr.at[ep_idx, step_idx].set(x),
            out_tree,
            current_data,
        )
        new_carry = (ep_idx, step_idx + 1, out_tree)
        new_carry = jax.lax.cond(
            rollout.terminations[idx] | rollout.truncations[idx],
            lambda c: (c[0] + 1, 0, c[2]),
            lambda c: c,
            new_carry,
        )
        return new_carry, None

    empty_tree = jax.tree.map(
        lambda x: jnp.full(
            (args.num_steps, args.num_steps) + x.shape[1:], jnp.nan, dtype=x.dtype
        ),
        rollout,
    )

    init_carry = (0, 0, empty_tree)
    (_, _, out_tree), _ = jax.lax.scan(body_fn, init_carry, jnp.arange(args.num_steps))

    return EpisodesRollout(**out_tree)


def collect_ppo_rollout(
    env: environment.Environment,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    episode_state: EpisodeState,
    training_state: TrainingState,
    args: PPOArguments,
    *,
    key: Key,
) -> tuple[EpisodeState, TrainingState, EpisodesRollout]:
    """Collect a PPO rollout from the environment and agent.

    Collects some number of episodes such that the total number of steps
    matches the number of steps in the PPO arguments.

    Returns:
    - episode_state: New episode state.
    - training_state: New training state.
    - episodes: Rollout of episodes.
    """
    episode_state = rollover_episode_state(episode_state)

    def scan_step(
        carry: tuple[EpisodeState, Key],
        _: None,
    ) -> tuple[tuple[EpisodeState, Key], EpisodesRollout]:
        """Wrapper for env_step to be used in jax.lax.scan."""
        episode_state, key = carry
        carry_key, step_key = jr.split(key, 2)

        episode_state, buffer = env_step(
            env, env_params, agent, episode_state, args, key=step_key
        )

        return (episode_state, carry_key), buffer

    (episode_state, _), rollout = jax.lax.scan(
        scan_step, (episode_state, key), length=args.num_steps
    )

    episodes = split_into_episodes(rollout, args)

    return episode_state, training_state, episodes


def compute_gae_episode(
    rewards: Float[Array, " num_steps"],
    values: Float[Array, " num_steps"],
    terminations: Bool[Array, " num_steps"],
    truncations: Bool[Array, " num_steps"],
    args: PPOArguments,
) -> tuple[Float[Array, " num_steps"], Float[Array, " num_steps"]]:
    """
    Compute GAE advantages and bootstrapped returns for an episode with nan padding.
    Assumes that valid (non‑nan) entries come first, followed by all nans.

    Arguments:
    - rewards: Rewards for the episode.
    - values: Values for the episode.
    - terminations: Terminations for the episode.
    - truncations: Truncations for the episode.
    - args: PPO arguments.

    Returns:
    - returns: Bootstrapped returns for the episode.
    - advantages: GAE advantages for the episode.
    """
    dones = terminations | truncations
    T = rewards.shape[0]
    num_valid = jnp.sum(~jnp.isnan(rewards)).astype(jnp.int64)
    idxs = jnp.arange(T)

    def scan_fn(
        carry: Float[Array, ""], t: Int[Array, ""]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        valid_idx = t < num_valid
        is_last_valid = t == (num_valid - 1)
        next_value = jax.lax.cond(
            is_last_valid,
            lambda: jnp.where(dones[t], 0.0, values[t]),
            lambda: values[t + 1],
        )
        delta = rewards[t] + args.gamma * next_value * (1.0 - dones[t]) - values[t]
        adv_next = jnp.where(is_last_valid, 0.0, carry)
        adv = delta + args.gamma * args.gae_lambda * adv_next * (1.0 - dones[t])
        adv = jnp.where(valid_idx, adv, jnp.nan)
        new_carry = jnp.where(valid_idx, adv, carry)
        return new_carry, adv

    rev_idxs = jnp.arange(T - 1, -1, -1)
    _, adv_rev = jax.lax.scan(scan_fn, jnp.array(0.0), rev_idxs)
    advantages = jnp.flip(adv_rev)
    returns = jnp.where(idxs < num_valid, advantages + values, jnp.nan)
    return returns, advantages


def compute_gae(episodes: EpisodesRollout, args: PPOArguments) -> EpisodesRollout:
    """
    Compute GAE advantages and bootstrapped returns for an EpisodesRollout.

    Args:
    - episodes: An EpisodesRollout with fields rewards, values, terminations, truncations.
        Note that `values` is shape [num_episodes, num_steps] (one value per step).
    - args: PPOArguments with gamma and gae_lambda.

    Returns:
    - episodes: An EpisodesRollout with advantage and return fields.
    """

    returns, advantages = jax.vmap(partial(compute_gae_episode, args=args))(
        episodes.rewards, episodes.values, episodes.terminations, episodes.truncations
    )

    return EpisodesRollout(
        observations=episodes.observations,
        actions=episodes.actions,
        log_probs=episodes.log_probs,
        entropies=episodes.entropies,
        values=episodes.values,
        rewards=episodes.rewards,
        terminations=episodes.terminations,
        truncations=episodes.truncations,
        advantages=advantages,
        returns=returns,
        times=episodes.times,
    )


def episode_loss(
    agent: CDEAgent,
    times: Float[Array, " N"],
    observations: Float[Array, " N observation_size"],
    actions: Float[Array, " N action_size"],
    log_probs: Float[Array, " N"],
    values: Float[Array, " N"],
    advantages: Float[Array, " N"],
    returns: Float[Array, " N"],
    args: PPOArguments,
) -> tuple[Float[Array, ""], PPOStats]:
    """Compute the PPO loss for a single episode.

    Safe to use on episodes with NaN padding.
    Returns NaN for the loss and stats if the episode is entirely NaN.

    Arguments:
    - agent: Agent to compute the loss for.
    - times: Time steps of the episode.
    - observations: Observations of the episode.
    - actions: Actions taken in the episode.
    - log_probs: Log probabilities of the actions.
    - values: Values of the states.
    - advantages: GAE advantages.
    - returns: Bootstrapped returns.
    - args: PPO arguments.

    Returns:
    - loss: PPO loss for the episode.
    - stats: PPO stats for the episode.
    """
    _, _, new_log_probs, new_entropies, new_values = agent.get_action_and_value(
        times, observations, actions, evolving_out=True
    )

    log_ratio = new_log_probs - log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = jnp.nanmean(ratio - log_ratio) - 1.0

    if args.normalize_advantage:
        advantages = (advantages - jnp.nanmean(advantages)) / (
            jnp.nanstd(advantages) + 1e-8
        )
    else:
        advantages = advantages

    policy_loss = -jnp.nanmean(
        jnp.minimum(
            advantages * ratio,
            advantages
            * jnp.clip(ratio, 1.0 - args.clip_coefficient, 1.0 + args.clip_coefficient),
        )
    )

    if args.clip_value_loss:
        clipped_values = values + jnp.clip(
            new_values - values, -args.clip_coefficient, args.clip_coefficient
        )
        value_loss = (
            jnp.nanmean(
                jnp.maximum(
                    jnp.square(new_values - returns),
                    jnp.square(clipped_values - returns),
                )
            )
            / 2.0
        )
    else:
        value_loss = jnp.nanmean(jnp.square(new_values - returns)) / 2.0

    entropy_loss = jnp.nanmean(new_entropies)
    total_loss = (
        policy_loss
        + args.value_coefficient * value_loss
        - args.entropy_coefficient * entropy_loss
    )

    stats = make_empty(
        PPOStats,
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        approx_kl=approx_kl,
    )

    return total_loss, stats


episode_grad = eqx.filter_value_and_grad(episode_loss, has_aux=True)


def empty_episode_loss(
    agent: CDEAgent,
) -> tuple[Float[Array, ""], PPOStats]:
    """Return a zero loss and stats for an empty episode."""
    loss = jnp.array(0.0)
    stats = PPOStats(
        total_loss=jnp.nan,
        policy_loss=jnp.nan,
        value_loss=jnp.nan,
        entropy_loss=jnp.nan,
        approx_kl=jnp.nan,
    )

    return loss, stats


empty_episode_grad = eqx.filter_value_and_grad(empty_episode_loss, has_aux=True)


def batch_grad(
    agent: CDEAgent, rollout: EpisodesRollout, args: PPOArguments
) -> tuple[Float[Array, " *N"], PPOStats, PyTree]:
    """Compute the PPO loss and gradient for a batch of episodes."""

    def scan_fn(carry, x):
        times, obs, actions, log_probs, values, adv, ret = x
        # Skip the episode if all times are NaN.
        (loss, stats), grad = jax.lax.cond(
            jnp.all(jnp.isnan(times)),
            lambda: empty_episode_grad(agent),
            lambda: episode_grad(
                agent,
                times,
                obs,
                actions,
                log_probs,
                values,
                adv,
                ret,
                args=args,
            ),
        )
        return carry, (loss, stats, grad)

    xs = (
        rollout.times,
        rollout.observations,
        rollout.actions,
        rollout.log_probs,
        rollout.values,
        rollout.advantages,
        rollout.returns,
    )

    _, (losses, stats_tree, grads) = jax.lax.scan(scan_fn, None, xs)
    total_loss = jnp.nanmean(losses)
    stats = jax.tree.map(jnp.nanmean, stats_tree)
    grads = jax.tree.map(jnp.nanmean, grads)

    return total_loss, stats, grads


def get_batch_indices(
    batch_size: int, dataset_size: int, *, key: Key | None = None
) -> Int[Array, " {dataset_size // batch_size} {batch_size}"]:
    """Get batch indices for a dataset of a given size."""
    indices = jnp.arange(dataset_size)

    if key is not None:
        indices = jax.random.permutation(key, indices)

    if dataset_size % batch_size != 0:
        logger.warning("Dataset size is not divisible by batch size.")

    indices = indices[: dataset_size - (dataset_size % batch_size)]
    return indices.reshape(-1, batch_size)


def train_on_batch(
    agent: CDEAgent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: EpisodesRollout,
    batch_indices: Int[Array, " N"],
    args: PPOArguments,
) -> tuple[CDEAgent, optax.OptState, PPOStats]:
    """Take a training step on a batch of episodes."""
    rollout = jax.tree.map(lambda x: x[batch_indices], rollout)
    loss, stats, grads = batch_grad(agent, rollout, args)

    any_nans = jnp.any(
        jnp.asarray(
            jax.tree.flatten(jax.tree.map(lambda l: jnp.any(jnp.isnan(l)), grads))[0]
        )
    )
    grads = eqx.error_if(grads, any_nans, "Gradients contain NaNs.")

    updates, opt_state = optimizer.update(grads, opt_state)
    agent = eqx.apply_updates(agent, updates)

    return agent, opt_state, stats


def epoch(
    agent: CDEAgent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: EpisodesRollout,
    args: PPOArguments,
    *,
    key: Key | None = None,
) -> tuple[CDEAgent, optax.OptState, PPOStats]:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    def scan_batch(
        carry: tuple[PyTree, optax.OptState], xs: Array
    ) -> tuple[tuple[PyTree, optax.OptState], PPOStats]:
        agent_dynamic, opt_state = carry
        agent = eqx.combine(agent_dynamic, agent_static)
        agent, opt_state, stats = train_on_batch(
            agent, optimizer, opt_state, rollout, xs, args
        )
        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)
        return (agent_dynamic, opt_state), stats

    indices = get_batch_indices(
        args.minibatch_size, rollout.observations.shape[0], key=key
    )
    (agent_dynamic, opt_state), stats = jax.lax.scan(
        scan_batch, (agent_dynamic, opt_state), indices
    )

    stats = jax.tree.map(jnp.nanmean, stats)
    agent = eqx.combine(agent_dynamic, agent_static)

    return agent, opt_state, stats


def train_on_rollout(
    agent: CDEAgent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: EpisodesRollout,
    args: PPOArguments,
    *,
    key: Key | None = None,
) -> tuple[CDEAgent, optax.OptState, PPOStats]:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    def scan_epoch(
        carry: tuple[PyTree, optax.OptState, Key | None],
        _: None,
    ) -> tuple[tuple[PyTree, optax.OptState, Key | None], PPOStats]:
        agent_dynamic, opt_state, key = carry
        agent = eqx.combine(agent_dynamic, agent_static)

        if key is not None:
            carry_key, batch_key = jr.split(key, 2)
        else:
            carry_key, batch_key = None, None

        agent, opt_state, stats = epoch(
            agent, optimizer, opt_state, rollout, args, key=batch_key
        )

        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)

        return (agent_dynamic, opt_state, carry_key), stats

    (agent_dynamic, opt_state, _), stats = jax.lax.scan(
        scan_epoch, (agent_dynamic, opt_state, key), length=args.num_epochs
    )

    stats = jax.tree.map(jnp.nanmean, stats)
    agent = eqx.combine(agent_dynamic, agent_static)

    return agent, opt_state, stats


@eqx.filter_jit
def train(
    env: environment.Environment,
    env_params: gym.EnvParams,
    agent: CDEAgent,
    args: PPOArguments,
    *,
    key: Key,
) -> CDEAgent:
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    if args.anneal_learning_rate:
        schedule = optax.schedules.cosine_decay_schedule(
            args.learning_rate,
            args.num_iterations
            * args.num_epochs
            * args.num_steps
            // args.minibatch_size,
        )
    else:
        schedule = optax.constant_schedule(args.learning_rate)

    adam = optax.inject_hyperparams(optax.adam)(learning_rate=schedule, eps=1e-5)
    optimizer = optax.named_chain(
        ("clipping", optax.clip_by_global_norm(args.max_gradient_norm)), ("adam", adam)
    )

    @eqx.filter_jit
    def scan_step(
        carry: tuple[TrainingState, EpisodeState, PyTree, optax.OptState, Key], _: None
    ) -> tuple[tuple[TrainingState, EpisodeState, PyTree, optax.OptState, Key], PyTree]:
        training_state, episode_state, agent_dynamic, opt_state, key = carry
        agent = eqx.combine(agent_dynamic, agent_static)
        carry_key, rollout_key, training_key = jr.split(key, 3)

        episode_state, training_state, rollout = collect_ppo_rollout(
            env, env_params, agent, episode_state, training_state, args, key=rollout_key
        )

        rollout = compute_gae(rollout, args)

        agent, opt_state, stats = train_on_rollout(
            agent, optimizer, opt_state, rollout, args, key=training_key
        )

        jax.debug.print("Stats:\n{}", stats, ordered=True)

        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)

        return (
            training_state,
            episode_state,
            agent_dynamic,
            opt_state,
            carry_key,
        ), stats

    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))
    reset_key, training_key = jr.split(key, 2)
    training_state = make_empty(
        TrainingState, global_step=jnp.array(0), opt_state=opt_state
    )
    episode_state = reset(env, env_params, args, key=reset_key)

    (training_state, episode_state, agent_dynamic, opt_state, _), _ = jax.lax.scan(
        scan_step,
        (training_state, episode_state, agent_dynamic, opt_state, training_key),
        length=args.num_iterations,
    )

    agent = eqx.combine(agent_dynamic, agent_static)

    return agent


if __name__ == "__main__":
    key: Key = jr.key(0)
    env, env_params = gym.make("Pendulum-v1")

    agent = CDEAgent(
        env=env,
        env_params=env_params,
        hidden_size=4,
        processed_size=4,
        width_size=128,
        depth=1,
        key=key,
    )

    args = PPOArguments(
        num_steps=1024,
        gamma=0.93,
        gae_lambda=0.95,
        num_epochs=16,
        normalize_advantage=True,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        max_gradient_norm=0.5,
        target_kl=None,
        minibatch_size=64,
        num_iterations=64,
        learning_rate=1e-3,
        anneal_learning_rate=True,
    )

    agent = train(env, env_params, agent, args, key=key)
