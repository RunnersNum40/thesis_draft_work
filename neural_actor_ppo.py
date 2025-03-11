import os
import random
import time
from dataclasses import dataclass

import equinox as eqx
import gymnasium as gym
import jax
import numpy as np
import optax
import tyro
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

from unbiased_neural_actor import UnbiasedNeuralActor


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    env_id: str = "Pendulum-v1"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    actor_timestep: float = 0.01
    """the timestep for the agent internal ODE"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    def __hash__(self):
        return hash(frozenset(vars(self).items()))


class Agent(eqx.Module):
    critic: eqx.nn.MLP
    actor_mean: UnbiasedNeuralActor
    actor_logstd: Array

    def __init__(self, env: gym.Env, *, key: Array, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        critic_key, actor_mean_key = jr.split(key)
        self.critic = eqx.nn.MLP(
            in_size=int(jnp.asarray(env.observation_space.shape).prod()),
            out_size=1,
            width_size=64,
            depth=1,
            activation=jax.nn.tanh,
            key=critic_key,
        )
        self.actor_mean = UnbiasedNeuralActor(
            state_shape=16,
            input_size=int(jnp.asarray(env.observation_space.shape).prod()),
            input_mapping_width=64,
            input_mapping_depth=2,
            output_size=int(jnp.asarray(env.action_space.shape).prod()),
            output_mapping_width=64,
            output_mapping_depth=2,
            key=actor_mean_key,
        )
        self.actor_logstd = jnp.zeros(jnp.asarray(env.action_space.shape).prod())

    @eqx.filter_jit
    def get_value(self, x: Array) -> Array:
        return self.critic(x).squeeze()

    @eqx.filter_jit
    def get_action_and_value(
        self, x: Array, state: Array, ts: Array, key: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        assert ts.shape == (2,)
        assert (
            state.shape == self.actor_mean.state_shape
        ), f"{state.shape=}, {self.actor_mean.state_shape=}"

        state, action_mean = self.actor_mean(ts, state, x, max_steps=2**14)
        assert action_mean is not None

        action_std = jnp.exp(self.actor_logstd)
        action = jr.normal(key, action_mean.shape) * action_std + action_mean
        logprob = jax.scipy.stats.norm.logpdf(action, action_mean, action_std)
        entropy = 0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1)

        return (
            state,
            action,
            logprob.sum(),
            entropy.sum(),
            self.get_value(x),
        )

    @eqx.filter_jit
    def get_action_value(
        self, x: Array, state: Array, ts: Array, action: Array
    ) -> tuple[Array, Array, Array]:
        assert ts.shape == (2,)
        assert state.shape == self.actor_mean.state_shape

        state, action_mean = self.actor_mean(ts, state, x, max_steps=2**14)
        assert action_mean is not None

        action_std = jnp.exp(self.actor_logstd)
        logprob = jax.scipy.stats.norm.logpdf(action, action_mean, action_std)
        entropy = 0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1)

        return logprob.sum(), entropy.sum(), self.get_value(x)

    def initial_state(self, key: Array) -> Array:
        return self.actor_mean.initial_state(key)


def compute_gae(
    agent: Agent,
    next_observation: Array,
    next_done: Array,
    rewards: Array,
    values: Array,
    dones: Array,
    args: Args,
) -> tuple[Array, Array]:
    next_value = agent.critic(next_observation)

    next_values = jnp.concatenate([values[1:], next_value], axis=0)
    next_non_terminal = jnp.concatenate(
        [1.0 - dones[1:], jnp.array([1.0 - next_done], dtype=rewards.dtype)], axis=0
    )

    def scan_fn(carry, x):
        reward, value, next_val, non_terminal = x
        delta = reward + args.gamma * next_val * non_terminal - value
        new_carry = delta + args.gamma * args.gae_lambda * non_terminal * carry
        return new_carry, new_carry

    inputs = (
        jnp.flip(rewards, axis=0),
        jnp.flip(values, axis=0),
        jnp.flip(next_values, axis=0),
        jnp.flip(next_non_terminal, axis=0),
    )
    inputs_stacked = jnp.stack(inputs, axis=1)
    init_carry = jnp.array(0.0, dtype=rewards.dtype)
    _, advantages_rev = jax.lax.scan(scan_fn, init_carry, inputs_stacked)

    advantages = jnp.flip(advantages_rev, axis=0)
    returns = advantages + values

    return advantages, returns


@eqx.filter_value_and_grad(has_aux=True)
def loss_grad(
    agent: Agent,
    observations: Array,
    states: Array,
    times: Array,
    actions: Array,
    advantages: Array,
    returns: Array,
    values: Array,
    log_probs: Array,
    args: Args,
) -> tuple[Array, dict[str, Array]]:
    new_log_prob, entropy, new_value = jax.vmap(agent.get_action_value)(
        observations, states, times, actions
    )
    log_ratio = new_log_prob - log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = jnp.mean(((ratio - 1) - log_ratio))

    if args.norm_adv:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    policy_loss = jnp.mean(
        jnp.maximum(
            -advantages * ratio,
            -advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
        )
    )

    if args.clip_vloss:
        value_loss_unclipped = (returns - new_value) ** 2
        value_clipped = values + jnp.clip(
            new_value - values, -args.clip_coef, args.clip_coef
        )
        value_loss_clipped = (returns - value_clipped) ** 2
        value_loss = (
            jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped)) / 2.0
        )
    else:
        value_loss = jnp.mean((returns - new_value) ** 2) / 2.0

    entropy_loss = jnp.mean(entropy)
    loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
    stats = {
        "total": loss,
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy_loss,
        "kl": approx_kl,
    }
    return loss, stats


def get_batch_indices(batch_size: int, data_size: int, key: Array) -> jnp.ndarray:
    perm = jr.permutation(key, data_size, independent=True)
    num_batches = data_size // batch_size
    return perm[: num_batches * batch_size].reshape(num_batches, batch_size)


@eqx.filter_jit
def training_step(
    agent: Agent,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    key: Array,
    next_observation: Array,
    next_done: Array,
    rewards: Array,
    dones: Array,
    observations: Array,
    states: Array,
    times: Array,
    actions: Array,
    values: Array,
    log_probs: Array,
    args: Args,
) -> tuple[Agent, optax.OptState, dict[str, Array]]:
    advantages, returns = compute_gae(
        agent, next_observation, next_done, rewards, values, dones, args
    )

    observations, states, times, actions, advantages, returns, values, log_probs = (
        jax.tree.map(
            jax.lax.stop_gradient,
            (
                observations,
                states,
                times,
                actions,
                advantages,
                returns,
                values,
                log_probs,
            ),
        )
    )

    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    def epoch(
        carry: tuple[Array, Array, optax.OptState], _
    ) -> tuple[tuple[Array, Array, optax.OptState], dict[str, Array]]:
        key, agent_dynamic, opt_state = carry

        def batch(
            carry: tuple[Array, optax.OptState], batch_indices: Array
        ) -> tuple[tuple[Array, optax.OptState], dict[str, Array]]:
            agent_dynamic, opt_state = carry
            agent: Agent = eqx.combine(agent_dynamic, agent_static)  # pyright: ignore

            (_, stats), grads = loss_grad(
                agent,
                observations[batch_indices],
                states[batch_indices],
                times[batch_indices],
                actions[batch_indices],
                advantages[batch_indices],
                returns[batch_indices],
                values[batch_indices],
                log_probs[batch_indices],
                args,
            )

            updates, opt_state = optimizer.update(grads, opt_state)
            agent = eqx.apply_updates(agent, updates)
            agent_dynamic, _ = eqx.partition(agent, eqx.is_array)
            return (agent_dynamic, opt_state), stats

        key, batch_key = jr.split(key)
        batch_indices = get_batch_indices(
            args.minibatch_size, args.batch_size, batch_key
        )

        (agent_dynamic, opt_state), stats = jax.lax.scan(
            batch,
            (agent_dynamic, opt_state),
            batch_indices,
        )
        stats = jax.tree.map(lambda x: x.mean(), stats)

        carry = (key, agent_dynamic, opt_state)
        return carry, stats

    init_carry = (key, agent_dynamic, opt_state)
    carry, stats = jax.lax.scan(epoch, init_carry, jnp.arange(args.update_epochs))
    stats = jax.tree.map(lambda x: x.mean(), stats)

    _, agent_dynamic, opt_state = carry
    agent = eqx.combine(agent_dynamic, agent_static)

    variance = jnp.var(returns)
    explained_variance = jnp.where(
        variance == 0, jnp.nan, 1 - jnp.var(returns - values) / variance
    )
    stats["explained_variance"] = explained_variance

    stats["learning_rate"] = opt_state[1].hyperparams["learning_rate"]

    return agent, opt_state, stats


def rollout(
    env: gym.Env,
    agent: Agent,
    agent_state: Array,
    agent_time: Array,
    next_observation: Array,
    next_done: Array,
    global_step: int,
    args: Args,
    writer: SummaryWriter,
    key: Array,
):
    observations = jnp.zeros((args.num_steps,) + env.observation_space.shape)
    states = jnp.zeros((args.num_steps,) + agent.actor_mean.state_shape)
    times = jnp.zeros((args.num_steps, 2))
    actions = jnp.zeros((args.num_steps,) + env.action_space.shape)
    log_probs = jnp.zeros((args.num_steps,))
    rewards = jnp.zeros((args.num_steps,))
    dones = jnp.zeros((args.num_steps,))
    values = jnp.zeros((args.num_steps,))

    for step in range(args.num_steps):
        global_step += 1

        ts = jnp.array([agent_time, agent_time + args.actor_timestep])
        observations = observations.at[step].set(next_observation)
        states = states.at[step].set(agent_state)
        times = times.at[step].set(ts)
        dones = dones.at[step].set(next_done)
        agent_time = ts[1]

        key, action_key = jr.split(key)
        agent_state, action, log_prob, _, value = agent.get_action_and_value(
            next_observation, agent_state, ts, action_key
        )
        values = values.at[step].set(value)
        actions = actions.at[step].set(action)
        log_probs = log_probs.at[step].set(log_prob)

        next_observation, reward, termination, truncation, info = env.step(action)
        rewards = rewards.at[step].set(reward)
        next_observation = jnp.asarray(next_observation)
        next_done = jnp.asarray(int(termination or truncation))

        if "episode" in info:
            writer.add_scalar("episode/reward", info["episode"]["r"], global_step)
            writer.add_scalar("episode/length", info["episode"]["l"], global_step)
            writer.add_scalar("episode/time", info["episode"]["t"], global_step)

        if termination or truncation:
            next_observation, _ = env.reset(seed=random.randint(0, 2**32 - 1))
            next_observation = jnp.array(next_observation)
            key, state_key = jr.split(key)
            agent_state = agent.initial_state(state_key)
            agent_time = jnp.array(0.0)

    return (
        agent_state,
        agent_time,
        next_observation,
        next_done,
        global_step,
        observations,
        states,
        times,
        actions,
        log_probs,
        rewards,
        dones,
        values,
    )


@eqx.filter_jit
def write_stats(
    writer: SummaryWriter, stats: dict[str, Array], global_step: int
) -> None:
    for stat, value in stats.items():
        jax.debug.callback(
            lambda stat, value, global_step: writer.add_scalar(
                f"loss/{stat}", float(value), global_step
            ),
            stat,
            value,
            global_step,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    print(f"Starting {run_name}")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jr.key(args.seed)

    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
    assert isinstance(
        env.observation_space, gym.spaces.Box
    ), "only continuous observation space is supported"
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    key, agent_key = jr.split(key)
    agent = Agent(env, key=agent_key)
    if args.anneal_lr:
        schedule = optax.schedules.cosine_decay_schedule(
            args.learning_rate,
            args.num_iterations * args.update_epochs * args.num_minibatches,
        )
    else:
        schedule = optax.constant_schedule(args.learning_rate)
    adam = optax.inject_hyperparams(optax.adam)(learning_rate=schedule)
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), adam)
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    global_step = 0
    next_done = jnp.array(0)

    next_observation, _ = env.reset(seed=random.randint(0, 2**32 - 1))
    next_observation = jnp.asarray(next_observation)
    key, state_key = jr.split(key)
    agent_state = agent.initial_state(state_key)
    agent_time = jnp.array(0.0)

    for iteration in trange(1, args.num_iterations + 1, desc="Training"):
        key, rollout_key = jr.split(key)
        (
            agent_state,
            agent_time,
            next_observation,
            next_done,
            global_step,
            observations,
            states,
            times,
            actions,
            log_probs,
            rewards,
            dones,
            values,
        ) = rollout(
            env,
            agent,
            agent_state,
            agent_time,
            next_observation,
            next_done,
            global_step,
            args,
            writer,
            rollout_key,
        )

        key, training_key = jr.split(key)
        agent, opt_state, stats = training_step(
            agent=agent,
            opt_state=opt_state,
            optimizer=optimizer,
            key=training_key,
            next_observation=next_observation,
            next_done=next_done,
            dones=dones,
            rewards=rewards,
            observations=observations,
            states=states,
            times=times,
            actions=actions,
            values=values,
            log_probs=log_probs,
            args=args,
        )

        write_stats(writer, stats, global_step)

    env.close()
    writer.close()
