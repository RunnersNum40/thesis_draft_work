import os
import random
import time
from dataclasses import dataclass
from typing import Generator

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

from tqdm_rich_without_warnings import trange


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    env_id: str = "InvertedPendulum-v5"
    """the id of the environment"""
    total_timesteps: int = 100000
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
    actor_mean: eqx.nn.MLP
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
            depth=3,
            activation=jax.nn.tanh,
            key=critic_key,
        )
        self.actor_mean = eqx.nn.MLP(
            in_size=int(jnp.asarray(env.observation_space.shape).prod()),
            out_size=int(jnp.asarray(env.action_space.shape).prod()),
            width_size=64,
            depth=3,
            activation=jax.nn.tanh,
            key=actor_mean_key,
        )
        self.actor_logstd = jnp.zeros(jnp.asarray(env.action_space.shape).prod())

    @eqx.filter_jit
    def get_value(self, x: Array) -> Array:
        return self.critic(x).squeeze()

    @eqx.filter_jit
    def get_action_and_value(
        self, x: Array, key: Array
    ) -> tuple[Array, Array, Array, Array]:
        action_mean = self.actor_mean(x)
        action_std = jnp.exp(self.actor_logstd)
        action = jr.normal(key, action_mean.shape) * action_std + action_mean
        logprob = jax.scipy.stats.norm.logpdf(action, action_mean, action_std)
        entropy = 0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1)

        return (
            action,
            logprob.sum(),
            entropy.sum(),
            self.get_value(x),
        )

    @eqx.filter_jit
    def get_action_value(self, x: Array, action: Array) -> tuple[Array, Array, Array]:
        action_mean = self.actor_mean(x)
        action_std = jnp.exp(self.actor_logstd)
        logprob = jax.scipy.stats.norm.logpdf(action, action_mean, action_std)
        entropy = 0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1)

        return logprob.sum(), entropy.sum(), self.get_value(x)


@eqx.filter_jit
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


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_grad(
    agent: Agent,
    observations: Array,
    actions: Array,
    advantages: Array,
    returns: Array,
    values: Array,
    log_probs: Array,
    args: Args,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    new_log_prob, entropy, new_value = jax.vmap(agent.get_action_value)(
        observations, actions
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
    return loss, (
        policy_loss,
        entropy_loss,
        value_loss,
        jax.lax.stop_gradient(approx_kl),
    )


@eqx.filter_jit
def train_batch(
    agent: Agent,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    observations: Array,
    actions: Array,
    advantages: Array,
    returns: Array,
    values: Array,
    log_probs: Array,
    args: Args,
) -> tuple[Array, Array, Array, Array, Array, Agent, optax.OptState]:
    observations, actions, advantages, returns, values, log_probs = jax.tree.map(
        jax.lax.stop_gradient,
        (observations, actions, advantages, returns, values, log_probs),
    )

    (loss, (policy_loss, value_loss, entropy_loss, approx_kl)), grads = loss_grad(
        agent, observations, actions, advantages, returns, values, log_probs, args
    )
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_agent = eqx.apply_updates(agent, updates)
    return (
        loss,
        policy_loss,
        value_loss,
        entropy_loss,
        approx_kl,
        new_agent,
        new_opt_state,
    )


@eqx.filter_jit
def get_batches(batch_size: int, data_size: int, key: Array) -> jnp.ndarray:
    perm = jr.permutation(key, data_size, independent=True)
    num_batches = data_size // batch_size
    return perm[: num_batches * batch_size].reshape(num_batches, batch_size)


def train_on_minibatch(
    agent: Agent,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    key: Array,
    observations: Array,
    actions: Array,
    advantages: Array,
    returns: Array,
    values: Array,
    log_probs: Array,
    args: Args,
) -> tuple[Array, Array, Array, Array, Array, Agent, optax.OptState]:
    losses = []
    policy_losses = []
    value_losses = []
    entropy_losses = []
    approx_kls = []
    for batch_indices in get_batches(args.minibatch_size, args.num_steps, key):
        (
            loss,
            policy_loss,
            value_loss,
            entropy_loss,
            approx_kl,
            agent,
            opt_state,
        ) = train_batch(
            agent,
            opt_state,
            optimizer,
            observations[batch_indices],
            actions[batch_indices],
            advantages[batch_indices],
            returns[batch_indices],
            values[batch_indices],
            log_probs[batch_indices],
            args,
        )
        losses.append(loss)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        entropy_losses.append(entropy_loss)
        approx_kls.append(approx_kl)

    loss = jnp.asarray(losses).mean()
    policy_loss = jnp.asarray(policy_losses).mean()
    value_loss = jnp.asarray(value_losses).mean()
    entropy_loss = jnp.asarray(entropy_losses).mean()
    approx_kl = jnp.asarray(approx_kls).mean()

    return loss, policy_loss, value_loss, entropy_loss, approx_kl, agent, opt_state


def update_step(
    agent: Agent,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    key: Array,
    observations: Array,
    actions: Array,
    advantages: Array,
    returns: Array,
    values: Array,
    log_probs: Array,
    args: Args,
) -> tuple[Agent, optax.OptState, Array, Array, Array, Array, Array, Array]:
    losses = []
    policy_losses = []
    value_losses = []
    entropy_losses = []
    approx_kls = []

    for _ in range(args.update_epochs):
        key, batch_key = jr.split(key)
        (
            loss,
            policy_loss,
            value_loss,
            entropy_loss,
            approx_kl,
            agent,
            opt_state,
        ) = train_on_minibatch(
            agent,
            opt_state,
            optimizer,
            batch_key,
            observations,
            actions,
            advantages,
            returns,
            values,
            log_probs,
            args,
        )
        losses.append(loss)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        entropy_losses.append(entropy_loss)
        approx_kls.append(approx_kl)

    loss = jnp.asarray(losses).mean()
    policy_loss = jnp.asarray(policy_losses).mean()
    value_loss = jnp.asarray(value_losses).mean()
    entropy_loss = jnp.asarray(entropy_losses).mean()
    approx_kl = jnp.asarray(approx_kls).mean()

    variance = jnp.var(returns)
    if variance == 0:
        explained_variance = jnp.array(0.0)
    else:
        explained_variance = 1 - jnp.var(returns - values) / variance

    return (
        agent,
        opt_state,
        loss,
        policy_loss,
        value_loss,
        entropy_loss,
        approx_kl,
        explained_variance,
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

    if args.capture_video:
        env = gym.make(args.env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(args.env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.clip(obs, -10, 10), env.observation_space
    )
    env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
    env = gym.wrappers.TransformReward(
        env, lambda reward: np.clip(float(reward), -10, 10)
    )
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
            args.learning_rate, args.total_timesteps
        )
    else:
        schedule = optax.constant_schedule(args.learning_rate)
    adam = optax.adam(learning_rate=schedule)
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), adam)
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    observations = jnp.zeros((args.num_steps,) + env.observation_space.shape)
    actions = jnp.zeros((args.num_steps,) + env.action_space.shape)
    log_probs = jnp.zeros((args.num_steps,))
    rewards = jnp.zeros((args.num_steps,))
    dones = jnp.zeros((args.num_steps,))
    values = jnp.zeros((args.num_steps,))

    global_step = 0
    next_observation, _ = env.reset(seed=args.seed)
    next_observation = jnp.asarray(next_observation)
    next_done = jnp.array(0)

    for iteration in trange(1, args.num_iterations + 1):
        for step in range(args.num_steps):
            global_step += 1
            observations = observations.at[step].set(next_observation)
            dones = dones.at[step].set(next_done)

            key, subkey = jr.split(key)
            action, log_prob, _, value = agent.get_action_and_value(
                next_observation, subkey
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
                next_observation, _ = env.reset(seed=args.seed)
                next_observation = jnp.array(next_observation)

        advantages, returns = compute_gae(
            agent, next_observation, next_done, rewards, values, dones, args
        )

        key, iteration_key = jr.split(key)
        (
            agent,
            opt_state,
            loss,
            policy_loss,
            value_loss,
            entropy_loss,
            approx_kl,
            explained_variance,
        ) = update_step(
            agent,
            opt_state,
            optimizer,
            iteration_key,
            observations,
            actions,
            advantages,
            returns,
            values,
            log_probs,
            args,
        )

        writer.add_scalar("loss/total", float(loss), global_step)
        writer.add_scalar("loss/policy", float(policy_loss), global_step)
        writer.add_scalar("loss/value", float(value_loss), global_step)
        writer.add_scalar("loss/entropy", float(entropy_loss), global_step)
        writer.add_scalar("loss/kl", float(approx_kl), global_step)
        writer.add_scalar(
            "loss/explained_variance", float(explained_variance), global_step
        )

    env.close()
    writer.close()
