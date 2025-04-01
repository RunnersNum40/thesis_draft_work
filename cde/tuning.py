import jax
import jax.numpy as jnp
import jax.random as jr
import optuna
import optunahub
import joblib
from jaxtyping import Key, Array

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
                action_key, step_key, carry_key = jr.split(key, 3)
                _, action, _, _, _ = agent.get_action_and_value(
                    ts=episode_state.times,
                    xs=episode_state.observations,
                    key=action_key,
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
    key = jr.key(trial.number)
    env, env_params = gym.make("Pendulum-v1")
    num_agents = 2
    total_steps = 524288

    weight_scale = trial.suggest_float("field_weight_scale", 0.1, 1.0, step=0.1)
    agent_timestep = trial.suggest_float("agent_timestep", 1e-2, 1.0, step=1e-2)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 32)
    clip_coefficient = trial.suggest_float("clip_coefficient", 0.1, 0.3, step=0.01)
    entropy_coefficient = trial.suggest_float(
        "entropy_coefficient", -0.05, 0.05, step=0.01
    )
    max_gradient_norm = trial.suggest_float("max_gradient_norm", 0.3, 1.0, step=0.01)
    num_steps = trial.suggest_categorical("num_steps", [1024, 2048, 4095])
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

    def test_agent(key: Key) -> Array:
        agent_key, train_key, eval_key = jr.split(key, 3)

        agent = CDEAgent(
            env=env,
            env_params=env_params,
            hidden_size=4,
            width_size=64,
            depth=2,
            weight_scale=weight_scale,
            key=agent_key,
        )
        agent = train(env, env_params, agent, args, train_key)
        scores = evaluate(agent, env, env_params, args, eval_key)

        return jnp.mean(scores)

    try:
        score = float(jnp.mean(jax.vmap(test_agent)(jr.split(key, num_agents))))
    except RuntimeError:
        score = float(-jnp.inf)
    return score


if __name__ == "__main__":
    n_workers = 8
    n_trials = 64

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
