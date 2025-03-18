import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import gymnax as gym
import jax
import optax
import tyro
from gymnax import wrappers
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from torch.utils.tensorboard.writer import SummaryWriter

from unbiased_neural_actor import UnbiasedNeuralActor
from utils import mlp_init

jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Agent(eqx.Module):
    critic: eqx.nn.MLP
    actor_mean: eqx.nn.MLP
    actor_logstd: Array
    preprocessing: UnbiasedNeuralActor

    def __init__(
        self,
        env: environment.Environment | GymnaxWrapper,
        env_params: environment.EnvParams,
        *,
        key: Array,
    ):
        state_size = 2

        preprocessing_key, critic_key, actor_mean_key = jr.split(key, 3)
        self.preprocessing = UnbiasedNeuralActor(
            state_shape=state_size,
            input_size=int(jnp.asarray(env.observation_space(env_params).shape).prod()),
            input_mapping_width=64,
            input_mapping_depth=0,
            output_size=state_size,
            output_mapping_width=64,
            output_mapping_depth=0,
            key=preprocessing_key,
        )
        self.critic = mlp_init(
            in_size=state_size,
            out_size="scalar",
            width_size=64,
            depth=2,
            final_std=1.0,
            key=critic_key,
        )
        self.actor_mean = mlp_init(
            in_size=state_size,
            out_size=int(jnp.asarray(env.action_space(env_params).shape).prod()),
            width_size=64,
            depth=2,
            final_std=0.01,
            key=actor_mean_key,
        )
        self.actor_logstd = jnp.zeros(
            jnp.asarray(env.action_space(env_params).shape).prod()
        )

    def get_value(self, x: Array, state: Array, ts: Array) -> Array:
        state, hidden = self.preprocessing(ts, state, x)
        return self.critic(hidden)

    def get_action_and_value(
        self, x: Array, state: Array, ts: Array, key: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        state, hidden = self.preprocessing(ts, state, x)
        action_mean = self.actor_mean(hidden)
        value = self.critic(hidden)

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
        state, hidden = self.preprocessing(ts, state, x)
        action_mean = self.actor_mean(hidden)
        value = self.critic(hidden)

        action_std = jnp.exp(self.actor_logstd)
        logprob = jnp.sum(jax.scipy.stats.norm.logpdf(action, action_mean, action_std))
        entropy = jnp.sum(0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1))

        return logprob, entropy, value

    def initial_state(self, x: Array, *, key: Array) -> Array:
        return self.preprocessing.initial_state(x, key=key)

    @property
    def state_shape(self):
        return self.preprocessing.state_shape


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    env_id: str = "Pendulum-v1"
    """the id of the environment"""
    total_timesteps: int = 1048576
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.93
    """the discount factor gamma"""
    gae_lambda: float = 0.96
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 16
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.29
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.04
    """coefficient of the entropy"""
    vf_coef: float = 0.8
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    actor_timestep: float = 1e-2
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


def collect_rollout(
    env: environment.Environment | GymnaxWrapper,
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
        next_observation, env_state = env.reset(env_state_key, env_params)
        agent_state = agent.initial_state(next_observation, key=agent_state_key)
        agent_time = jnp.array(0.0)
        # Fix the LogWrapper typing
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
        return next_observation, env_state, agent_state, agent_time

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
        clipped_action = jnp.clip(
            action, env.action_space(env_params).low, env.action_space(env_params).high
        )
        key, env_step_key = jr.split(key)
        next_observation, env_state, reward, next_done, info = env.step(
            env_step_key, env_state, clipped_action, env_params
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


def get_batch_indices(batch_size: int, data_size: int, key: Array) -> jnp.ndarray:
    perm = jr.permutation(key, data_size, independent=True)
    num_batches = data_size // batch_size
    return perm[: num_batches * batch_size].reshape(num_batches, batch_size)


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
    advantages, returns = compute_gae(
        training_states["next_done"],
        next_value,
        rollouts["rewards"],
        rollouts["values"],
        rollouts["dones"],
        args,
    )
    rollouts["advantages"] = advantages
    rollouts["returns"] = returns

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
                observations=rollouts["observations"][batch_indices],
                states=rollouts["states"][batch_indices],
                times=rollouts["times"][batch_indices],
                actions=rollouts["actions"][batch_indices],
                advantages=rollouts["advantages"][batch_indices],
                returns=rollouts["returns"][batch_indices],
                values=rollouts["values"][batch_indices],
                log_probs=rollouts["log_probs"][batch_indices],
                args=args,
            )

            updates, opt_state = optimizer.update(grads, opt_state)

            stats["grads"] = jnp.linalg.norm(
                jnp.asarray(jax.tree.flatten(jax.tree.map(jnp.linalg.norm, grads))[0])
            )
            stats["updates"] = jnp.linalg.norm(
                jnp.asarray(jax.tree.flatten(jax.tree.map(jnp.linalg.norm, updates))[0])
            )

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

    stats["learning_rate"] = opt_state["adam"].hyperparams["learning_rate"]

    return agent, opt_state, stats


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
    env: environment.Environment | GymnaxWrapper,
    env_params: environment.EnvParams,
    agent: Agent,
    args,
    key: Array,
    max_steps: int = 1000,
) -> tuple[Array, Array]:
    key, env_state_key, agent_state_key, key = jr.split(key, 4)
    next_observation, env_state = env.reset(env_state_key, env_params)
    agent_state = agent.initial_state(next_observation, key=agent_state_key)

    state_storage = jnp.zeros((max_steps, *agent_state.shape))

    state = {
        "key": key,
        "env_state": env_state,
        "agent_state": agent_state,
        "agent_time": jnp.array(0.0),
        "next_obs": next_observation,
        "total_reward": jnp.array(0.0),
        "done": jnp.array(False),
        "steps": jnp.array(0),
        "state_storage": state_storage,
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

        steps = state["steps"]
        state_storage = state["state_storage"].at[steps].set(new_agent_state)

        return {
            "key": new_key,
            "env_state": new_env_state,
            "agent_state": new_agent_state,
            "agent_time": ts[1],
            "next_obs": next_obs,
            "total_reward": state["total_reward"] + reward,
            "done": done,
            "steps": steps + 1,
            "state_storage": state_storage,
        }

    final_state = jax.lax.while_loop(not_complete, env_step, state)
    recorded_states = final_state["state_storage"][: final_state["steps"]]

    return final_state["total_reward"], recorded_states


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    key = jr.key(args.seed)

    env, env_params = gym.make(args.env_id)
    env = wrappers.LogWrapper(env)
    env = TransformObservationWrapper(
        env,
        lambda obs, params, key: jnp.clip(obs[:2], -10.0, 10.0),
        spaces.Box(-10.0, 10.0, (2,), jnp.float64),
    )
    env = TransformRewardWrapper(
        env, lambda reward, params, key: jnp.clip(reward, -10.0, 10.0)
    )

    key, agent_key = jr.split(key)
    agent = Agent(env, env_params, key=agent_key)
    agent_dynamic, agent_static = eqx.partition(agent, eqx.is_array)

    if args.anneal_lr:
        schedule = optax.schedules.cosine_decay_schedule(
            args.learning_rate,
            args.num_iterations * args.update_epochs * args.num_minibatches,
        )
    else:
        schedule = optax.constant_schedule(args.learning_rate)
    adam = optax.inject_hyperparams(optax.adam)(learning_rate=schedule, eps=1e-5)
    optimizer = optax.named_chain(
        ("clipping", optax.clip_by_global_norm(args.max_grad_norm)), ("adam", adam)
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    key, env_key = jr.split(key)
    next_observation, env_state = env.reset(env_key, env_params)
    key, state_key = jr.split(key)

    key, rollout_key = jr.split(key)
    training_states = {
        "key": rollout_key,
        "env_state": env_state,
        "agent_state": agent.initial_state(next_observation, key=state_key),
        "agent_time": jnp.array(0.0),
        "next_observation": next_observation,
        "next_done": jnp.array(False),
        "global_step": jnp.array(0),
    }

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    @eqx.filter_jit
    def train_step(
        carry: tuple[Array, optax.OptState, dict[str, Array], Array], _
    ) -> tuple[tuple[Array, optax.OptState, dict[str, Array], Array], None]:
        agent_dynamic, opt_state, training_state, key = carry
        agent: Agent = eqx.combine(agent_dynamic, agent_static)  # pyright: ignore

        training_state, rollout = collect_rollout(
            env=env,
            env_params=env_params,
            agent=agent,
            training_states=training_state,
            args=args,
            writer=writer,
        )

        key, training_key = jr.split(key)
        agent, opt_state, stats = train_on_rollout(
            agent=agent,
            opt_state=opt_state,
            optimizer=optimizer,
            training_states=training_state,
            rollouts=rollout,
            args=args,
            key=training_key,
        )

        write_stats(writer, logger, stats, training_state["global_step"])

        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)
        return (agent_dynamic, opt_state, training_state, key), None

    init_carry = (agent_dynamic, opt_state, training_states, key)
    final_carry, _ = jax.lax.scan(train_step, init_carry, length=args.num_iterations)

    agent_dynamic, opt_state, training_states, key = final_carry
    agent = eqx.combine(agent_dynamic, agent_static)

    writer.close()

    key, eval_key = jr.split(key)
    scores, states = jax.vmap(evaluate, in_axes=(None, None, None, None, 0))(
        env, env_params, agent, args, jr.split(eval_key, 10)
    )
    logger.info(f"Score: {scores.mean()}")

    import numpy as np
    from matplotlib import pyplot as plt

    states = np.array(states[0])
    times = np.arange(states.shape[0]) * args.actor_timestep

    plt.plot(times, states)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.show()
