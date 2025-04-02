import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import joblib
import optuna
import optunahub
from jaxtyping import Array, Key

from main import (
    CDEAgent,
    PPOArguments,
    gym,
    reset_episode,
    train,
)


def evaluate(
    agent: CDEAgent,
    env,
    env_params,
    args: PPOArguments,
    key: Key,
    num_eval_episodes: int = 5,
) -> Array:
    def run_episode(key: Key):
        key, reset_key = jr.split(key)
        episode_state = reset_episode(env, env_params, args, key=reset_key)

        def step_fn(carry, _):
            step, episode_state, cum_reward, key, done_flag = carry

            def do_step():
                step_key, carry_key = jr.split(key, 2)
                # Deterministic actions
                _, action, _, _, _ = agent.get_action_and_value(
                    ts=episode_state.times,
                    xs=episode_state.observations,
                )
                clipped_action = jnp.clip(
                    action,
                    env.action_space(env_params).low,
                    env.action_space(env_params).high,
                )
                obs, new_env_state, reward, done, _ = env.step(
                    step_key, episode_state.env_state, clipped_action, env_params
                )
                new_episode_state = episode_state.replace(
                    step=episode_state.step + 1,
                    env_state=new_env_state,
                    observations=episode_state.observations.at[
                        episode_state.step + 1
                    ].set(obs),
                    times=episode_state.times.at[episode_state.step + 1].set(
                        episode_state.times[episode_state.step] + args.agent_timestep
                    ),
                )
                return (
                    step + 1,
                    new_episode_state,
                    cum_reward + reward,
                    carry_key,
                    done,
                )

            new_carry = jax.lax.cond(
                done_flag,
                lambda: (step, episode_state, cum_reward, key, done_flag),
                do_step,
            )
            return new_carry, None

        init_carry = (0, episode_state, 0.0, key, False)
        final_carry, _ = jax.lax.scan(step_fn, init_carry, jnp.arange(args.num_steps))
        return final_carry[2]

    rewards = jax.vmap(run_episode)(jr.split(key, num_eval_episodes))
    return jnp.mean(rewards)


def objective(trial: optuna.Trial) -> float:
    # Model parameters
    hidden_size = 3
    width_size = 64
    depth = 2
    actor_output_final_activation = {"tanh": jnn.tanh, "relu": jnn.relu}[
        trial.suggest_categorical("actor_output_final_activation", ["tanh", "relu"])
    ]
    const_entropy = trial.suggest_categorical("const_entropy", [True, False])

    # Gradient application
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 32)

    # Gradient calculation
    clip_coefficient = trial.suggest_float("clip_coefficient", 0.1, 0.3)
    state_coefficient = trial.suggest_float("state_coefficient", 0.0, 1.0)
    max_gradient_norm = trial.suggest_float("max_gradient_norm", 0.0, 2.0)

    # Batching
    total_steps = 524288
    num_steps = trial.suggest_categorical("num_steps", [1024, 2048, 4095])
    num_minibatches = trial.suggest_categorical("minibatch_size", [4, 8, 16, 32, 64])
    minibatch_size = num_steps // num_minibatches
    num_batches = trial.suggest_categorical("num_batches", [1, 2, 4, 8, 16])
    batch_size = num_minibatches // num_batches

    key = jr.key(trial.number)
    env, env_params = gym.make("Pendulum-v1")

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
        state_coefficient=state_coefficient,
        max_gradient_norm=max_gradient_norm,
        anneal_learning_rate=False,
        tb_logging=False,
    )

    def test_agent(key: Key) -> Array:
        agent_key, train_key, eval_key = jr.split(key, 3)

        agent = CDEAgent(
            env=env,
            env_params=env_params,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=agent_key,
            const_std=const_entropy,
            actor_output_final_activation=actor_output_final_activation,
        )
        agent = train(env, env_params, agent, args, train_key)
        scores = evaluate(agent, env, env_params, args, eval_key)

        return jnp.mean(scores)

    try:
        num_agents = 2
        score = float(jnp.mean(jax.vmap(test_agent)(jr.split(key, num_agents))))
    except RuntimeError:
        score = float(jnp.nan)
    return score


if __name__ == "__main__":
    n_workers = 16
    n_trials = 128

    def create_study() -> optuna.Study:
        module = optunahub.load_module(package="samplers/auto_sampler")
        study = optuna.create_study(
            direction="maximize",
            study_name="CDEAgent-exhaustive",
            load_if_exists=True,
            storage="sqlite:///cde_agent.db",
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
    print("Best trial:")
    print(study.best_params)
