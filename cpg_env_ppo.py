import os
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import cpg_env

NUM_ENVS = 4
EPISODE_STEPS = 500
EVAL_STEPS = 500
TIMESTEPS = 1e6
SAVE_INTERVAL = 1e5
SAVE_DIR = "checkpoints"
LOG_DIR = "/tmp/tensorboard"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.zip")


def make_env(env_id: str, *args, **kwargs) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(env_id, *args, **kwargs)
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env

    return _init


env_fn = make_env(
    "EllipseCPGEnv-v0",
    time_limit=EPISODE_STEPS,
    n=1,
    state_noise=0.1,
    observation_noise=0.01,
    action_noise=0.01,
    observe_actions=1,
)
eval_env_fn = make_env(
    "EllipseCPGEnv-v0",
    time_limit=EVAL_STEPS,
    n=1,
    state_noise=0.0,
    observation_noise=0.0,
    action_noise=0.0,
    observe_actions=1,
)


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    env_fns = [env_fn] * NUM_ENVS

    vec_env = SubprocVecEnv(env_fns, start_method="fork")
    eval_env = env_fn()

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_INTERVAL // NUM_ENVS,
        save_path=SAVE_DIR,
        name_prefix="ppo_cpg",
        save_vecnormalize=True,
    )
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=3,
        min_evals=5,
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        eval_freq=SAVE_INTERVAL // NUM_ENVS,
        callback_after_eval=stop_callback,
        verbose=1,
    )

    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from {BEST_MODEL_PATH}")
        model = PPO.load(BEST_MODEL_PATH, env=vec_env, tensorboard_log=LOG_DIR)
    else:
        print("No previous best model found, initializing a new model.")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            tensorboard_log=LOG_DIR,
        )

    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True,
        callback=[checkpoint_callback, eval_callback],
    )
