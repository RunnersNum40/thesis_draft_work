import logging

import gymnasium as gym
import numpy as np
from tqdm.rich import tqdm

import cpg_env
import sdf
from cpg import CPGParams, CPGState
from visualization import animate_trajectory, plot_polar_trajectory, plot_trajectory

logging.basicConfig(level=logging.INFO)


class MatchAmplitude:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.dt = env.unwrapped.dt

    @property
    def state(self) -> CPGState:
        return self.env.unwrapped.state

    def predict(self, obs: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, dict]:
        phase = obs[0][0]
        amplitude = obs[1][0]

        if self.env.unwrapped.shape == "square":
            intrinsic_amplitude = amplitude - sdf.square(
                amplitude * np.array([np.cos(phase), np.sin(phase)]),
                self.env.unwrapped.half_size,
            )
        elif self.env.unwrapped.shape == "ellipse":
            intrinsic_amplitude = amplitude - sdf.ellipse(
                amplitude * np.array([np.cos(phase), np.sin(phase)]),
                self.env.unwrapped.a,
                self.env.unwrapped.b,
            )
        else:
            raise ValueError(f"Unknown shape: {self.env.unwrapped.shape}")

        return np.array([[intrinsic_amplitude], [np.pi / 2.0]]), {}


def rollout_run(env: gym.Env, model, steps: int = 1000, seed: int = 0):
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

    logging.info(f"Total reward: {sum(rewards)}")

    return sum(rewards), states, params


def visualize_run(
    env: gym.Env,
    states: list[CPGState],
    params: list[CPGParams],
    show_params: bool = True,
):
    states_and_params = list(zip(states, params))
    plot_trajectory(states_and_params, env.unwrapped.dt * 10, show_params=show_params)
    plot_polar_trajectory(states_and_params, show_params=show_params)
    animate_trajectory(states_and_params)


if __name__ == "__main__":
    env = gym.make(
        "SquareCPGEnv-v0",
        n=1,
        state_noise=0.1,
        observation_noise=0.01,
        action_noise=0.01,
        observe_actions=1,
    )
    model = MatchAmplitude(env)
    reward, states, params = rollout_run(env, model, 1000, 1)
    print(f"Reward {reward}")
    visualize_run(env, states, params)
