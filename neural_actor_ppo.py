import logging
import os
import time
from dataclasses import dataclass

import equinox as eqx
import gymnax as gym
import jax
import optax
import tyro
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from gymnax import wrappers
from gymnax.environments import environment, spaces

from unbiased_neural_actor import UnbiasedNeuralActor
from utils import mlp_with_final_layer_std

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    actor_mean: eqx.nn.MLP
    actor_logstd: Array
    preprocessing: UnbiasedNeuralActor

    def __init__(
        self,
        env: environment.Environment | wrappers.LogWrapper,
        env_params: environment.EnvParams,
        *,
        key: Array,
    ):
        state_size = 4

        preprocessing_key, critic_key, actor_mean_key = jr.split(key, 3)
        self.preprocessing = UnbiasedNeuralActor(
            state_shape=state_size,
            input_size=int(jnp.asarray(env.observation_space(env_params).shape).prod()),
            input_mapping_width=64,
            input_mapping_depth=3,
            output_size=state_size,
            output_mapping_width=0,
            output_mapping_depth=0,
            key=preprocessing_key,
        )
        self.critic = mlp_with_final_layer_std(
            in_size=state_size,
            out_size="scalar",
            width_size=64,
            depth=3,
            std=1.0,
            activation=jax.nn.tanh,
            key=critic_key,
        )
        self.actor_mean = mlp_with_final_layer_std(
            in_size=state_size,
            out_size=int(jnp.asarray(env.action_space(env_params).shape).prod()),
            width_size=64,
            depth=3,
            std=0.01,
            activation=jax.nn.tanh,
            key=actor_mean_key,
        )
        self.actor_logstd = jnp.zeros(
            jnp.asarray(env.action_space(env_params).shape).prod()
        )

    @eqx.filter_jit
    def get_value(self, x: Array, state: Array, ts: Array) -> Array:
        state, x = self.preprocessing(ts, state, x, max_steps=2**14)
        return self.critic(x)

    @eqx.filter_jit
    def get_action_and_value(
        self, x: Array, state: Array, ts: Array, key: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        state, x = self.preprocessing(ts, state, x, max_steps=2**14)
        action_mean = self.actor_mean(x)
        value = self.critic(x)

        action_std = jnp.exp(self.actor_logstd)
        action = jr.normal(key, action_mean.shape) * action_std + action_mean
        logprob = jnp.sum(jax.scipy.stats.norm.logpdf(action, action_mean, action_std))
        entropy = jnp.sum(0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1))

        return (
            state,
            action,
            logprob,
            entropy,
            value,
        )

    @eqx.filter_jit
    def get_action_value(
        self, x: Array, state: Array, ts: Array, action: Array
    ) -> tuple[Array, Array, Array]:
        state, x = self.preprocessing(ts, state, x, max_steps=2**14)
        action_mean = self.actor_mean(x)
        value = self.critic(x)

        action_std = jnp.exp(self.actor_logstd)
        logprob = jnp.sum(jax.scipy.stats.norm.logpdf(action, action_mean, action_std))
        entropy = jnp.sum(0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1))

        return logprob, entropy, value

    def initial_state(self, key: Array) -> Array:
        return self.preprocessing.initial_state(key)

    @property
    def state_shape(self):
        return self.preprocessing.state_shape


def compute_gae(
    next_done: Array,
    next_value: Array,
    rewards: Array,
    values: Array,
    dones: Array,
    args: Args,
) -> tuple[Array, Array]:
    next_values = jnp.concatenate([values[1:], jnp.expand_dims(next_value, 0)], axis=0)
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
    next_value = agent.get_value(next_observation, states[-1], times[-1])
    advantages, returns = compute_gae(
        next_done, next_value, rewards, values, dones, args
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


def log_info(writer: SummaryWriter, info: dict[str, Array], global_step: Array) -> None:
    r = info["returned_episode_returns"]
    l = info["returned_episode_lengths"]
    jax.debug.callback(
        lambda r, l: logger.debug(f"Episode finished with reward {r} and length {l}"),
        r,
        l,
        ordered=True,
    )
    jax.debug.callback(
        lambda r, global_step: writer.add_scalar(
            "episode/reward", float(r), int(global_step)
        ),
        r,
        global_step,
    )
    jax.debug.callback(
        lambda l, global_step: writer.add_scalar(
            "episode/length", float(l), int(global_step)
        ),
        l,
        global_step,
    )


@eqx.filter_jit
def rollout(
    env: environment.Environment | wrappers.LogWrapper,
    env_params: environment.EnvParams,
    env_state: Array,
    agent: Agent,
    agent_state: Array,
    agent_time: Array,
    next_observation: Array,
    next_done: Array,
    global_step: Array,
    args: Args,
    writer: SummaryWriter,
    key: Array,
):
    def log_info(info: dict[str, Array], global_step: Array) -> None:
        r = info["returned_episode_returns"]
        l = info["returned_episode_lengths"]
        jax.debug.callback(
            lambda r, l: logger.debug(
                f"Episode finished with reward {r} and length {l}"
            ),
            r,
            l,
            ordered=True,
        )
        jax.debug.callback(
            lambda r, global_step: writer.add_scalar(
                "episode/reward", float(r), int(global_step)
            ),
            r,
            global_step,
        )
        jax.debug.callback(
            lambda l, global_step: writer.add_scalar(
                "episode/length", float(l), int(global_step)
            ),
            l,
            global_step,
        )

    def reset(key: Array, env_state: Array) -> tuple[Array, Array, Array, Array]:
        key, env_state_key = jr.split(key)
        next_observation, env_state = env.reset(env_state_key, env_params)

        key, agent_state_key = jr.split(key)
        agent_state = agent.initial_state(agent_state_key)
        agent_time = jnp.array(0.0)

        # Fix the types of the episode returns
        env_state = eqx.tree_at(
            lambda s: s.episode_returns,
            env_state,
            replace_fn=lambda x: x.astype(jnp.float64),
        )
        env_state = eqx.tree_at(
            lambda s: s.returned_episode_returns,
            env_state,
            replace_fn=lambda x: x.astype(jnp.float64),
        )

        return next_observation, env_state, agent_state, agent_time

    def rollout_step(carry, _):
        key, env_state, agent_state, agent_time, next_obs, next_done, global_step = (
            carry
        )
        global_step += 1
        ts = jnp.array([agent_time, agent_time + args.actor_timestep])
        out_obs = next_obs
        out_state = agent_state
        out_time = ts

        key, action_key = jr.split(key)
        agent_state_new, action, log_prob, _, value = agent.get_action_and_value(
            next_obs, agent_state, ts, action_key
        )
        action = jnp.clip(
            action, env.action_space(env_params).low, env.action_space(env_params).high
        )

        key, env_step_key = jr.split(key)
        next_obs_new, env_state_new, reward, next_done_new, info = env.step(
            env_step_key, env_state, action, env_params
        )

        key, _ = jr.split(key)
        jax.lax.cond(
            info["returned_episode"],
            lambda _: log_info(info, global_step),
            lambda _: None,
            operand=None,
        )

        new_agent_time = ts[1]
        key, reset_key = jr.split(key)
        next_obs_final, env_state_final, agent_state_final, agent_time_final = (
            jax.lax.cond(
                next_done_new,
                lambda _: reset(reset_key, env_state_new),
                lambda _: (
                    next_obs_new,
                    env_state_new,
                    agent_state_new,
                    new_agent_time,
                ),
                operand=None,
            )
        )
        next_done_final = jax.lax.cond(
            next_done_new,
            lambda _: jnp.array(False),
            lambda _: next_done_new,
            operand=None,
        )

        new_carry = (
            key,
            env_state_final,
            agent_state_final,
            agent_time_final,
            next_obs_final,
            next_done_final,
            global_step,
        )
        step_output = (
            out_obs,
            out_state,
            out_time,
            action,
            log_prob,
            reward,
            next_done_new,
            value,
        )
        return new_carry, step_output

    initial_carry = (
        key,
        env_state,
        agent_state,
        agent_time,
        next_observation,
        next_done,
        global_step,
    )
    carry, outputs = jax.lax.scan(
        rollout_step, initial_carry, None, length=args.num_steps
    )
    observations, states, times, actions, log_probs, rewards, dones, values = outputs
    (
        key,
        env_state,
        agent_state,
        agent_time,
        next_observation,
        next_done,
        global_step,
    ) = carry
    return (
        env_state,
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
    writer: SummaryWriter, stats: dict[str, Array], global_step: Array
) -> None:
    for stat, value in stats.items():
        jax.debug.callback(
            lambda stat, value, global_step: writer.add_scalar(
                f"loss/{stat}", float(value), int(global_step)
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

    key = jr.key(args.seed)

    env, env_params = gym.make(args.env_id)
    env = wrappers.LogWrapper(env)

    key, agent_key = jr.split(key)
    agent = Agent(env, env_params, key=agent_key)
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

    global_step = jnp.array(0)
    next_done = jnp.array(False)

    key, env_key = jr.split(key)
    next_observation, env_state = env.reset(env_key, env_params)
    key, state_key = jr.split(key)
    agent_state = agent.initial_state(state_key)
    agent_time = jnp.array(0.0)

    with logging_redirect_tqdm():
        for iteration in trange(1, args.num_iterations + 1, desc="Training"):
            key, rollout_key = jr.split(key)
            (
                env_state,
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
                env_params,
                env_state,
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
