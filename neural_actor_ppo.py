import logging
import os
import time
from dataclasses import dataclass

import equinox as eqx
import gymnax as gym
import jax
import optax
import tyro
from gymnax import wrappers
from gymnax.environments import environment
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

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

    def get_value(self, x: Array, state: Array, ts: Array) -> Array:
        state, x = self.preprocessing(ts, state, x, max_steps=2**14)
        return self.critic(x)

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


def training_step(
    agent: Agent,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    key: Array,
    training_states: dict[str, Array],
    rollouts: dict[str, Array],
    args: Args,
) -> tuple[Agent, optax.OptState, dict[str, Array]]:
    next_value = agent.get_value(
        training_states["next_observation"],
        training_states["agent_state"],
        jnp.array(
            [
                training_states["agent_time"],
                training_states["agent_time"] + args.actor_timestep,
            ]
        ),
    )
    _advantages, _returns = compute_gae(
        training_states["next_done"],
        next_value,
        rollouts["rewards"],
        rollouts["values"],
        rollouts["dones"],
        args,
    )
    rollouts["advantages"] = _advantages
    rollouts["returns"] = _returns

    rollouts = jax.tree.map(jax.lax.stop_gradient, rollouts)

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
                rollouts["observations"][batch_indices],
                rollouts["states"][batch_indices],
                rollouts["times"][batch_indices],
                rollouts["actions"][batch_indices],
                rollouts["advantages"][batch_indices],
                rollouts["returns"][batch_indices],
                rollouts["values"][batch_indices],
                rollouts["log_probs"][batch_indices],
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

    variance = jnp.var(rollouts["returns"])
    explained_variance = jnp.where(
        variance == 0,
        jnp.nan,
        1 - jnp.var(rollouts["returns"] - rollouts["values"]) / variance,
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


def rollout(
    env: environment.Environment | wrappers.LogWrapper,
    env_params: environment.EnvParams,
    agent: Agent,
    training_states: dict[str, Array],
    args: Args,
    writer: SummaryWriter,
) -> tuple[dict[str, Array], dict[str, Array]]:
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
        env_state_key, agent_state_key = jr.split(key)
        next_obs, env_state = env.reset(env_state_key, env_params)
        agent_state = agent.initial_state(agent_state_key)
        agent_time = jnp.array(0.0)
        # Fix the state typing
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
        return next_obs, env_state, agent_state, agent_time

    def rollout_step(carry, _):
        key = carry["key"]
        env_state = carry["env_state"]
        agent_state = carry["agent_state"]
        agent_time = carry["agent_time"]
        next_observation = carry["next_observation"]
        global_step = carry["global_step"]

        global_step += 1
        ts = jnp.array([agent_time, agent_time + args.actor_timestep])
        out_observation = next_observation
        out_state = agent_state
        out_time = ts

        key, action_key = jr.split(key)
        agent_state, action, log_prob, _, value = agent.get_action_and_value(
            next_observation, agent_state, ts, action_key
        )
        action = jnp.clip(
            action, env.action_space(env_params).low, env.action_space(env_params).high
        )

        key, env_step_key = jr.split(key)
        next_observation, env_state, reward, next_done, info = env.step(
            env_step_key, env_state, action, env_params
        )

        key, _ = jr.split(key)
        jax.lax.cond(
            info["returned_episode"],
            lambda _: log_info(info, global_step),
            lambda _: None,
            operand=None,
        )

        agent_time = ts[1]
        key, reset_key = jr.split(key)
        next_observation, env_state, agent_state, agent_time = jax.lax.cond(
            next_done,
            lambda _: reset(reset_key, env_state),
            lambda _: (
                next_observation,
                env_state,
                agent_state,
                agent_time,
            ),
            operand=None,
        )

        carry = {
            "key": key,
            "env_state": env_state,
            "agent_state": agent_state,
            "agent_time": agent_time,
            "next_observation": next_observation,
            "next_done": next_done,
            "global_step": global_step,
        }
        output = {
            "observations": out_observation,
            "states": out_state,
            "times": out_time,
            "actions": action,
            "log_probs": log_prob,
            "rewards": reward,
            "dones": next_done,
            "values": value,
        }
        return carry, output

    training_states, rollouts = jax.lax.scan(
        rollout_step, training_states, None, length=args.num_steps
    )
    return training_states, rollouts


def write_stats(
    writer: SummaryWriter,
    logger: logging.Logger,
    stats: dict[str, Array],
    global_step: Array,
) -> None:
    jax.debug.callback(
        lambda global_step: logger.debug(f"Step {global_step}"),
        global_step,
        ordered=True,
    )
    for stat, value in stats.items():
        jax.debug.callback(
            lambda stat, value: logger.debug(f"{stat}: {value}"),
            stat,
            value,
            ordered=True,
        )
        jax.debug.callback(
            lambda stat, value, global_step: writer.add_scalar(
                f"loss/{stat}", float(value), int(global_step)
            ),
            stat,
            value,
            global_step,
            ordered=True,
        )


@eqx.filter_jit
def iteration(
    env: environment.Environment | wrappers.LogWrapper,
    env_params: environment.EnvParams,
    agent: Agent,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    training_states: dict[str, Array],
    key: Array,
) -> tuple[Agent, optax.OptState, dict[str, Array]]:
    training_states, rollouts = rollout(
        env=env,
        env_params=env_params,
        agent=agent,
        training_states=training_states,
        args=args,
        writer=writer,
    )

    agent, opt_state, stats = training_step(
        agent=agent,
        opt_state=opt_state,
        optimizer=optimizer,
        training_states=training_states,
        rollouts=rollouts,
        args=args,
        key=key,
    )

    write_stats(writer, logger, stats, training_states["global_step"])

    return (
        agent,
        opt_state,
        training_states,
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

    key, env_key = jr.split(key)
    next_observation, env_state = env.reset(env_key, env_params)
    key, state_key = jr.split(key)

    key, rollout_key = jr.split(key)
    training_states = {
        "key": rollout_key,
        "env_state": env_state,
        "agent_state": agent.initial_state(state_key),
        "agent_time": jnp.array(0.0),
        "next_observation": next_observation,
        "next_done": jnp.array(False),
        "global_step": jnp.array(0),
    }

    with logging_redirect_tqdm():
        for _ in trange(1, args.num_iterations + 1, desc="Training"):
            key, iteration_key = jr.split(key)
            agent, opt_state, training_states = iteration(
                env=env,
                env_params=env_params,
                agent=agent,
                optimizer=optimizer,
                opt_state=opt_state,
                training_states=training_states,
                key=iteration_key,
            )

    writer.close()
