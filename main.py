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

from neural_actor_ppo import (
    Agent,
    collect_rollout,
    evaluate,
    train_on_rollout,
    write_stats,
)

jax.config.update("jax_enable_x64", True)

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
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 846
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.93
    """the discount factor gamma"""
    gae_lambda: float = 0.96
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 22
    """the number of mini-batches"""
    update_epochs: int = 16
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.29
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.04
    """coefficient of the entropy"""
    vf_coef: float = 0.8
    """coefficient of the value function"""
    max_grad_norm: float = 0.23
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    actor_timestep: float = 1e-3
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
        obs, reward, done, info, state = self._env.step(env_key, state, action, params)
        return self.func(obs, params, wrapper_key), reward, done, info, state


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    key = jr.key(args.seed)

    env, env_params = gym.make(args.env_id)
    env = wrappers.LogWrapper(env)
    env = TransformObservationWrapper(
        env,
        lambda obs, params, key: obs[:2],
        spaces.Box(-1.0, 1.0, (2,), jnp.float64),
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
    adam = optax.inject_hyperparams(optax.adam)(learning_rate=schedule)
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), adam)
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    key, env_key = jr.split(key)
    next_observation, env_state = env.reset(env_key, env_params)
    print(next_observation)
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
    score = jnp.mean(
        jax.vmap(evaluate, in_axes=(None, None, None, None, 0))(
            env, env_params, agent, args, jr.split(eval_key, 10)
        )
    )
    logger.info(f"Score: {score}")
