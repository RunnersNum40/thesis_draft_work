import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from tqdm.rich import tqdm

import cpg_env
from visualization import animate_trajectory, plot_polar_trajectory, plot_trajectory

BEST_MODEL_PATH = "checkpoints/best_model.zip"
EPISODE_STEPS = 600

env = gym.make("EllipseCPGEnv-v0", n=1, disturbance=1e-3)
model = PPO.load(BEST_MODEL_PATH)


def visualize_run(
    env: gym.Env, model: PPO, steps: int = int(2 * np.pi * 100), seed: int = 0
):
    obs, _ = env.reset(seed=seed)
    states = []
    params = []
    rewards = []

    for _ in tqdm(range(steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, termination, truncation, _ = env.step(action)
        states.append(env.unwrapped.state)
        params.append(cpg_env.action_to_params(env.unwrapped.previous_action))
        rewards.append(reward)

        if termination or truncation:
            break

    print(f"Total reward: {np.sum(rewards)}")

    states_and_params = list(zip(states, params))
    plot_trajectory(states_and_params, env.unwrapped.dt)
    plot_polar_trajectory(states_and_params)
    animate_trajectory(states_and_params)


visualize_run(env, model, EPISODE_STEPS, 0)
