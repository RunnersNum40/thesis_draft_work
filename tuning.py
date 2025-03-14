import logging
import os
from dataclasses import dataclass

import equinox as eqx
import gymnax as gym
import jax
import optax
import optuna
from gymnax import wrappers
from jax import Array
from jax import numpy as jnp
from jax import random as jr

from neural_actor_ppo import Agent, collect_rollout, evaluate, train_on_rollout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    env_id: str = "Pendulum-v1"
    total_timesteps: int = 100000

    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    actor_timestep: float = 0.01

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    def __hash__(self):
        return hash(frozenset(vars(self).items()))


def test(args: Args) -> float:
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    key = jr.key(args.seed)

    env, env_params = gym.make(args.env_id)

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
        )

        key, training_key = jr.split(key)
        agent, opt_state, _ = train_on_rollout(
            agent=agent,
            opt_state=opt_state,
            optimizer=optimizer,
            training_states=training_state,
            rollouts=rollout,
            args=args,
            key=training_key,
        )

        agent_dynamic, _ = eqx.partition(agent, eqx.is_array)
        return (agent_dynamic, opt_state, training_state, key), None

    init_carry = (agent_dynamic, opt_state, training_states, key)
    final_carry, _ = jax.lax.scan(train_step, init_carry, length=args.num_iterations)

    agent_dynamic, opt_state, training_states, key = final_carry
    agent = eqx.combine(agent_dynamic, agent_static)

    key, eval_key = jr.split(key)
    score = jnp.mean(
        jax.vmap(evaluate, in_axes=(None, None, None, None, 0))(
            env, env_params, agent, args, jr.split(eval_key, 10)
        )
    )

    return float(score)


def objective(trial: optuna.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    num_steps = trial.suggest_int("num_steps", 128, 4096, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    num_minibatches = trial.suggest_int("num_minibatches", 1, 64)
    update_epochs = trial.suggest_int("update_epochs", 1, 20)
    norm_adv = trial.suggest_categorical("norm_adv", [True, False])
    clip_coef = trial.suggest_float("clip_coef", 0.1, 0.3)
    clip_vloss = trial.suggest_categorical("clip_vloss", [True, False])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 0.9)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 1.0)
    actor_timestep = trial.suggest_float("actor_timestep", 0.01, 0.1)

    args = Args(
        learning_rate=learning_rate,
        num_steps=num_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        norm_adv=norm_adv,
        clip_coef=clip_coef,
        clip_vloss=clip_vloss,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        actor_timestep=actor_timestep,
    )

    return test(args)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=4)
    study.trials_dataframe().to_csv(f"{os.path.basename(__file__)}-{Args.env_id}.csv")

    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")
