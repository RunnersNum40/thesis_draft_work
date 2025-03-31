import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optuna
import optunahub
from jaxtyping import Key

from main import (
    CDEAgent,
    PPOArguments,
    gym,
    reset_episode,
    train,
)

train = eqx.filter_jit(train)


def evaluate(
    agent: CDEAgent,
    env,
    env_params,
    args: PPOArguments,
    key: Key,
    num_eval_episodes: int = 5,
) -> float:
    """Run agent for a few episodes and return average reward."""
    total_reward = 0.0
    for _ in range(num_eval_episodes):
        eval_key, key = jr.split(key)
        episode_state = reset_episode(env, env_params, args, key=eval_key)
        rewards = []

        for t in range(args.num_steps):
            _, action, _, _, _ = agent.get_action_and_value(
                ts=episode_state.times,
                xs=episode_state.observations,
                key=key,
            )

            clipped_action = jnp.clip(
                action,
                env.action_space(env_params).low,
                env.action_space(env_params).high,
            )
            obs, env_state, reward, done, _ = env.step(
                key, episode_state.env_state, clipped_action, env_params
            )
            rewards.append(reward)

            episode_state = episode_state.replace(
                step=episode_state.step + 1,
                env_state=env_state,
                observations=episode_state.observations.at[episode_state.step + 1].set(
                    obs
                ),
                times=episode_state.times.at[episode_state.step + 1].set(
                    episode_state.times[episode_state.step] + args.agent_timestep
                ),
            )

            if done:
                break

        total_reward += jnp.sum(jnp.array(rewards))

    return float(total_reward) / num_eval_episodes


def objective(trial: optuna.Trial) -> float:
    key = jr.key(trial.number)
    env, env_params = gym.make("Pendulum-v1")

    # === CDE / Model-Specific ===
    weight_scale = trial.suggest_float("field_weight_scale", 0.1, 1.0, step=0.1)

    # === PPO ===
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 32)
    clip_coefficient = trial.suggest_float("clip_coefficient", 0.1, 0.3, step=0.01)
    entropy_coefficient = trial.suggest_float(
        "entropy_coefficient", -0.05, 0.05, step=0.01
    )
    max_gradient_norm = trial.suggest_float("max_gradient_norm", 0.3, 1.0, step=0.01)

    # === Sequence / Minibatch ===
    total_steps = 524288
    num_steps = trial.suggest_categorical("num_steps", [512, 1024, 2048, 4095])
    agent_timestep = trial.suggest_float("agent_timestep", 1e-2, 1.0, step=0.01)
    minibatch_size = trial.suggest_categorical("minibatch_size", [4, 8, 16, 32])
    num_minibatches = num_steps // minibatch_size
    num_batches = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    batch_size = num_minibatches // num_batches

    args = PPOArguments(
        run_name=f"tune-{trial.number}",
        num_iterations=total_steps // num_steps,
        num_steps=num_steps,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        num_minibatches=num_minibatches,
        batch_size=batch_size,
        num_batches=num_batches,
        learning_rate=learning_rate,
        clip_coefficient=clip_coefficient,
        entropy_coefficient=entropy_coefficient,
        max_gradient_norm=max_gradient_norm,
        agent_timestep=agent_timestep,
        anneal_learning_rate=False,
        tb_logging=False,
    )

    agent = CDEAgent(
        env=env,
        env_params=env_params,
        hidden_size=4,
        width_size=64,
        depth=2,
        weight_scale=weight_scale,
        key=key,
    )

    agent = train(env, env_params, agent, args, key)
    score = evaluate(agent, env, env_params, args, key)
    return score


if __name__ == "__main__":
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(
        direction="maximize",
        study_name="CDEAgent-exhaustive",
        load_if_exists=True,
        storage="sqlite:///cde_agent.db",
        sampler=module.AutoSampler(),
    )
    study.optimize(objective, n_trials=32, timeout=3600, n_jobs=4)

    print("Best trial:")
    print(study.best_params)
