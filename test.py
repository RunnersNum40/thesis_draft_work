import gymnasium as gym

import numpy as np
import cpg_env
from cpg import CPGState
from visualization import animate_trajectory, plot_polar_trajectory, plot_trajectory

EPISODE_STEPS = 1000

env = gym.make("SquareCPGEnv-v0", max_episode_steps=EPISODE_STEPS, n=1)


class MatchAmplitude:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.dt = env.unwrapped.dt
        self.half_size = env.unwrapped.half_size

    @property
    def state(self) -> CPGState:
        return self.env.unwrapped.state

    def predict(self, obs: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, dict]:
        phase = self.state.phase[0] % np.pi

        amplitude = self.half_size / np.max(np.abs([np.cos(phase), np.sin(phase)]))
        frequency = 1.0

        return np.array([[amplitude], [frequency]]), {}


model = MatchAmplitude(env)


def visualize_run(env, model, steps=1000):
    obs, _ = env.reset()
    states = []
    params = []
    rewards = []

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, termination, truncation, _ = env.step(action)
        states.append(env.unwrapped.state)
        params.append(cpg_env.action_to_params(action))
        rewards.append(reward)

        if termination or truncation:
            break

    print(f"Total reward: {np.sum(rewards)}")

    states_and_params = list(zip(states, params))
    plot_trajectory(states_and_params, env.unwrapped.dt)
    plot_polar_trajectory(states_and_params)
    animate_trajectory(states_and_params)


visualize_run(env, model, EPISODE_STEPS)
