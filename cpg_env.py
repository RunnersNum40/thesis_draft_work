from typing import Optional

import gymnasium as gym
import numpy as np

from cpg import CPGParams, CPGState, step_cpg
import sdf


def action_to_params(
    action: np.ndarray,
    convergence_factor: float = 100.0,
) -> CPGParams:
    n = action.shape[1]

    return CPGParams(
        intrinsic_amplitude=action[0],
        intrinsic_frequency=action[1],
        convergence_factor=np.repeat(convergence_factor, n),
        coupling_strength=np.zeros((n, n)),
        phase_bias=np.zeros((n, n)),
    )


def state_to_observation(state: CPGState) -> np.ndarray:
    return np.array([state.phase, state.amplitude])


def state_to_output(state: CPGState) -> np.ndarray:
    return state.amplitude * np.array([np.cos(state.phase), np.sin(state.phase)])


class CPGEnv(gym.Env):
    distance_weight = 1.0
    velocity_weight = 1.0

    def __init__(
        self,
        n: int = 1,
        time_limit: Optional[int] = None,
        dt: float = 1e-3,
        state_noise: float = 0.0,
        action_noise: float = 0.0,
        observation_noise: float = 0.0,
        time_step_range: tuple[int, int] = (1, 21),
        observe_actions: int = 1,
    ) -> None:
        self.n = n
        self.time_limit = time_limit
        self.dt = dt
        self.state_noise = state_noise
        self.action_noise = action_noise
        self.observation_noise = observation_noise
        self.time_step_range = time_step_range
        self.observe_actions = observe_actions

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                2,
                n + n + self.observe_actions * n,
            ),  # state, output, previous actions
            dtype=np.float64,
        )

        action_low = np.array([0.0, np.pi])
        action_high = np.array([2.0, 2 * np.pi])

        self.action_space = gym.spaces.Box(
            low=np.repeat([action_low], n, axis=0).T,
            high=np.repeat([action_high], n, axis=0).T,
            shape=(2, n),
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        state_observation = state_to_observation(self.state)
        action_observation = np.concatenate(self.previous_actions, axis=1)
        output_observation = state_to_output(self.state)

        observation = np.concatenate(
            [state_observation, output_observation, action_observation], axis=1
        )

        return observation + self.np_random.normal(
            0.0, self.observation_noise, observation.shape
        )

    def _get_reward(self) -> float:
        angular_velocity = self._angular_velocity()
        distance = self._get_distance()

        return (
            float(
                angular_velocity * self.velocity_weight
                - distance * self.distance_weight
            )
            * self.dt
        )

    def _get_terminated(self) -> bool:
        return False

    def _get_truncated(self) -> bool:
        return self.time_limit is not None and self.step_count >= self.time_limit

    def _get_info(self) -> dict:
        return {}

    @property
    def previous_action(self) -> np.ndarray:
        return self.previous_actions[-1]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.step_count = 0

        self.previous_actions = [
            self.action_space.sample() for _ in range(self.observe_actions)
        ]

        self.previous_state = CPGState(
            self.np_random.uniform(-np.pi, np.pi, self.n),
            self.np_random.uniform(0.0, 2.0, self.n),
            self.np_random.uniform(-1.0, 1.0, self.n),
        )
        self.state = step_cpg(
            self.previous_state, action_to_params(self.previous_action), self.dt
        )

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        params = action_to_params(
            action + self.np_random.normal(0.0, self.action_noise, action.shape)
        )

        self.previous_actions = self.previous_actions[1:] + [action]
        self.previous_state = self.state
        self.step_count += 1

        self.state = step_cpg(
            self.state,
            params,
            self.dt * self.np_random.integers(*self.time_step_range, endpoint=True),
        )

        if self.state_noise > 0.0:
            self.state = CPGState(
                self.state.phase
                + self.np_random.normal(0.0, self.state_noise, self.state.phase.shape)
                * self.dt,
                self.state.amplitude
                + self.np_random.normal(
                    0.0, self.state_noise, self.state.amplitude.shape
                )
                * self.dt,
                self.state.amplitude_dot
                + self.np_random.normal(
                    0.0, self.state_noise, self.state.amplitude_dot.shape
                )
                * self.dt,
            )

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _angular_velocity(self) -> float:
        if self.previous_state is None:
            return 0.0

        return self.state.phase[0] - self.previous_state.phase[0] / self.dt

    def _get_distance(self) -> float | np.ndarray:
        raise NotImplementedError


class SquareCPGEnv(CPGEnv):
    distance_weight = 1.0
    velocity_weight = 0.0

    half_size = 1.0
    shape = "square"

    def _get_distance(self) -> float | np.ndarray:
        x = self.state.amplitude[0] * np.cos(self.state.phase[0])
        y = self.state.amplitude[0] * np.sin(self.state.phase[0])
        return abs(sdf.square(np.array([x, y]), self.half_size))


gym.register(
    id="SquareCPGEnv-v0",
    entry_point=lambda *args, **kwargs: SquareCPGEnv(*args, **kwargs),
)


class EllipseCPGEnv(CPGEnv):
    distance_weight = 1.0
    velocity_weight = 0.0

    a = 1.0
    b = 0.5

    shape = "ellipse"

    def _get_distance(self) -> float | np.ndarray:
        x = self.state.amplitude[0] * np.cos(self.state.phase[0])
        y = self.state.amplitude[0] * np.sin(self.state.phase[0])
        return abs(sdf.ellipse(np.array([x, y]), self.a, self.b))


gym.register(
    id="EllipseCPGEnv-v0",
    entry_point=lambda *args, **kwargs: EllipseCPGEnv(*args, **kwargs),
)


if __name__ == "__main__":
    from visualization import animate_trajectory, plot_polar_trajectory, plot_trajectory

    env = gym.make("EllipseCPGEnv-v0", n=2)
    obs, _ = env.reset()

    NUM_TIMESTEPS = 1000
    states = [env.unwrapped.state]
    params = []

    for _ in range(NUM_TIMESTEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(env.unwrapped.state)
        params.append(action_to_params(action))

    states_and_params = list(zip(states, params))
    plot_trajectory(states_and_params, env.unwrapped.dt)
    plot_polar_trajectory(states_and_params)
    animate_trajectory(states_and_params)
