import logging

import equinox as eqx
import jax
import optax
from gymnax import wrappers
from gymnax.environments import environment
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from torch.utils.tensorboard.writer import SummaryWriter

from unbiased_neural_actor import UnbiasedNeuralActor
from utils import mlp_with_final_layer_std

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    args,
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
    args,
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


def train_on_rollout(
    agent: Agent,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    key: Array,
    training_states: dict[str, Array],
    rollouts: dict[str, Array],
    args,
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


def collect_rollout(
    env: environment.Environment | wrappers.LogWrapper,
    env_params: environment.EnvParams,
    agent: Agent,
    training_states: dict[str, Array],
    args,
    writer: SummaryWriter | None = None,
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
        if isinstance(env, wrappers.LogWrapper):
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
        if writer is not None:
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


def evaluate(
    env: environment.Environment | wrappers.LogWrapper,
    env_params: environment.EnvParams,
    agent: Agent,
    args,
    key: Array,
    max_steps: int = 1000,
) -> Array:
    key, env_state_key, agent_state_key, key = jr.split(key, 4)
    next_obs, env_state = env.reset(env_state_key, env_params)
    agent_state = agent.initial_state(agent_state_key)

    state = {
        "key": key,
        "env_state": env_state,
        "agent_state": agent_state,
        "agent_time": jnp.array(0.0),
        "next_obs": next_obs,
        "total_reward": jnp.array(0.0),
        "done": jnp.array(False),
        "steps": jnp.array(0),
    }

    def not_complete(state: dict[str, Array]):
        return jnp.logical_and(
            state["steps"] < max_steps, jnp.logical_not(state["done"])
        )

    def env_step(state: dict[str, Array]):
        keys = jr.split(state["key"], 3)
        new_key, action_key, step_key = keys[0], keys[1], keys[2]
        ts = jnp.array([state["agent_time"], state["agent_time"] + args.actor_timestep])
        new_agent_state, action, _, _, _ = agent.get_action_and_value(
            state["next_obs"], state["agent_state"], ts, action_key
        )
        action = jnp.clip(
            action,
            env.action_space(env_params).low,
            env.action_space(env_params).high,
        )
        next_obs, new_env_state, reward, done, _ = env.step(
            step_key, state["env_state"], action, env_params
        )
        return {
            "key": new_key,
            "env_state": new_env_state,
            "agent_state": new_agent_state,
            "agent_time": ts[1],
            "next_obs": next_obs,
            "total_reward": state["total_reward"] + reward,
            "done": done,
            "steps": state["steps"] + 1,
        }

    final_state = jax.lax.while_loop(not_complete, env_step, state)
    return final_state["total_reward"]
