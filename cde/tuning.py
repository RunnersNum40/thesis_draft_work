import optuna
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jaxtyping import Key

from main import (
    gym,
    train,
    CDEAgent,
    PPOArguments,
    reset_episode,
)


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

    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    entropy_coef = trial.suggest_float("entropy_coefficient", 0.0, 0.05)
    value_coef = trial.suggest_float("value_coefficient", 0.1, 1.0)
    clip_coef = trial.suggest_float("clip_coefficient", 0.1, 0.3)
    agent_timestep = trial.suggest_float("agent_timestep", 0.001, 1.0, log=True)
    field_activation = trial.suggest_categorical(
        "field_activation", ["tanh", "softplus"]
    )

    args = PPOArguments(
        run_name=f"tune-{trial.number}",
        num_iterations=64,
        num_steps=1024,
        num_epochs=16,
        num_minibatches=8,
        minibatch_size=32,
        num_batches=4,
        batch_size=2,
        agent_timestep=agent_timestep,
        learning_rate=lr,
        entropy_coefficient=entropy_coef,
        value_coefficient=value_coef,
        clip_coefficient=clip_coef,
        anneal_learning_rate=False,
        tb_logging=False,
    )

    agent = CDEAgent(
        env=env,
        env_params=env_params,
        hidden_size=4,
        processed_size=4,
        width_size=8,
        depth=1,
        key=key,
        actor_depth=0,
        output_depth=0,
        critic_depth=1,
        field_activation={"tanh": jnn.tanh, "softplus": jnn.softplus}[field_activation],
    )

    agent = train(env, env_params, agent, args, key)
    score = evaluate(agent, env, env_params, args, key)
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="CDEAgent",
        load_if_exists=True,
        storage="sqlite:///cde_agent.db",
    )
    study.optimize(objective, n_trials=100, timeout=1800, n_jobs=2)

    print("Best trial:")
    print(study.best_params)
