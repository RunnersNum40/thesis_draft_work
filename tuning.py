import equinox as eqx
import jax
import jax.nn as jnn
from jax import lax
import jax.numpy as jnp
import jax.random as jr
import joblib
import optuna
import optunahub
from jaxtyping import Array, Key, Float

from ppo import PPO, Env, gym
import wrappers
from policies import SharedNeuralCDEActorCriticPolicy


def evaluate(
    policy: PPO,
    state: eqx.nn.State,
    env: Env,
    env_params: gym.EnvParams,
    key: Key,
    num_eval_episodes: int = 5,
    max_episode_length: int = 500,
) -> Float[Array, ""]:
    def run_episode(key: Key) -> Float[Array, ""]:
        def step_fn(carry, _):
            env_state, policy_state, obs, done, total_reward, step_key = carry

            step_key, carry_key = jr.split(step_key)

            action, policy_state = policy.policy(state).predict_action(
                obs, policy_state
            )
            obs, env_state, reward, done_step, _ = env.step(
                step_key, env_state, action, env_params
            )

            total_reward += reward
            done = jnp.logical_or(done, done_step)

            return (env_state, policy_state, obs, done, total_reward, carry_key), done

        reset_key, step_key = jr.split(key)
        obs, env_state = env.reset(reset_key, env_params)
        policy_state = policy.policy(state).reset(
            env_state, env_params, eqx.nn.State({})
        )
        total_reward = jnp.array(0.0)
        done = jnp.array(False)

        (env_state, policy_state, obs, done, total_reward, _), _ = lax.scan(
            step_fn,
            (env_state, policy_state, obs, done, total_reward, step_key),
            None,
            length=max_episode_length,
        )

        return total_reward

    episode_keys = jr.split(key, num_eval_episodes)
    rewards = jax.vmap(run_episode)(episode_keys)
    return jnp.mean(rewards)


def objective(trial: optuna.Trial) -> float:
    key = jr.key(trial.number)

    # Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    num_steps = trial.suggest_int("num_steps", 1024, 8192, step=1024)
    num_epochs = trial.suggest_int("num_epochs", 1, 10)
    num_minibatches = trial.suggest_int("num_minibatches", 1, 64)

    clip_coefficient = trial.suggest_float("clip_coefficient", 0.1, 0.3)
    entropy_coefficient = trial.suggest_float("entropy_coefficient", 0.0, 0.02)
    value_coefficient = trial.suggest_float("value_coefficient", 0.25, 1.0)
    state_coefficient = trial.suggest_float("state_coefficient", 0.0, 1.0)

    # Model architecture
    width_size = trial.suggest_categorical("width_size", [32, 64, 128, 256])
    depth = trial.suggest_int("depth", 1, 4)
    state_size = trial.suggest_categorical("state_size", [8, 16, 32, 64])
    num_features = trial.suggest_categorical("num_features", [4, 8, 16, 32])
    max_steps = trial.suggest_int("max_steps", 2, 64, log=True)

    env_id = "Pendulum-v1"
    env, env_params = gym.make(env_id)
    env = wrappers.ClipActionWrapper(env)
    env = wrappers.RescaleAction(env)
    env = wrappers.AddTimeWrapper(env)

    ppo_agent, state = eqx.nn.make_with_state(PPO)(
        policy_class=SharedNeuralCDEActorCriticPolicy,
        policy_args=(),
        policy_kwargs={
            "width_size": width_size,
            "depth": depth,
            "state_size": state_size,
            "num_features": num_features,
            "max_steps": max_steps,
        },
        env=env,
        env_params=env_params,
        learning_rate=learning_rate,
        anneal_learning_rate=True,
        num_steps=num_steps,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        clip_coefficient=clip_coefficient,
        entropy_coefficient=entropy_coefficient,
        value_coefficient=value_coefficient,
        state_coefficient=state_coefficient,
        key=key,
    )

    total_timesteps = 524288
    state = ppo_agent.learn(
        state,
        total_timesteps=total_timesteps,
        key=key,
    )

    eval_key = jr.split(key)[0]
    avg_reward = evaluate(
        ppo_agent, state, env, env_params, eval_key, num_eval_episodes=5
    )

    return float(avg_reward)


if __name__ == "__main__":
    n_workers = 8
    n_trials = 128

    def create_study() -> optuna.Study:
        module = optunahub.load_module(package="samplers/auto_sampler")
        study = optuna.create_study(
            direction="maximize",
            study_name="CDEAgent-exhaustive",
            load_if_exists=True,
            storage="sqlite:///tuning.db",
            sampler=module.AutoSampler(),
        )
        return study

    def optimize_study():
        study = create_study()
        study.optimize(objective, n_trials=n_trials // n_workers, n_jobs=1)

    study = create_study()
    joblib.Parallel(n_workers)(
        joblib.delayed(optimize_study)() for _ in range(n_workers)
    )
