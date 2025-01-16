from typing import Optional
import numpy as np
import gymnasium as gym
from cpg import CPGParams, CPGState, step_cpg


def action_to_params(
    action: np.ndarray, convergence_factor: float = 100.0
) -> CPGParams:
    n = action.shape[1]

    intrinsic_amplitude = action[0]
    intrinsic_frequency = action[1]
    coupling_strength = np.zeros((n, n))
    phase_bias = np.zeros((n, n))

    return CPGParams(
        intrinsic_amplitude=intrinsic_amplitude,
        intrinsic_frequency=intrinsic_frequency,
        convergence_factor=np.repeat(convergence_factor, n),
        coupling_strength=coupling_strength,
        phase_bias=phase_bias,
    )


def state_to_observation(state: CPGState) -> np.ndarray:
    return np.array([state.phase, state.amplitude])


class CPGEnv(gym.Env):
    distance_weight = 1.0e-2
    velocity_weight = 0.0

    def __init__(
        self, n: int, dt: float = 1e-2, observe_last_action: bool = True
    ) -> None:
        self.n = n
        self.dt = dt
        self.observe_last_action = observe_last_action

        action_space_low = np.array([np.array(0.0).repeat(n), np.array(1.0).repeat(n)])
        action_space_high = np.array(
            [np.array(2.0).repeat(n), np.array(10.0).repeat(n)]
        )

        if self.observe_last_action:
            observation_space_shape = (2 + 2, n)
        else:
            observation_space_shape = (2, n)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_space_shape,
            dtype=np.float64,
        )

        self.action_space = gym.spaces.Box(
            low=action_space_low,
            high=action_space_high,
            shape=(2, n),
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        state_observation = state_to_observation(self.state)
        if self.observe_last_action:
            if self.previous_action is None:
                action_observation = np.zeros((2, self.n))
            else:
                action_observation = self.previous_action

            return np.concatenate([state_observation, action_observation])

        return state_observation

    def _get_reward(self) -> float:
        angular_velocity = self._angular_velocity()
        distance = self._get_distance()

        return angular_velocity * self.velocity_weight - distance * self.distance_weight

    def _get_terminated(self) -> bool:
        return False

    def _get_truncated(self) -> bool:
        return False

    def _get_info(self) -> dict:
        return {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.previous_state: Optional[CPGState] = None
        self.previous_action: Optional[np.ndarray] = None

        self.state = CPGState(
            np.zeros(self.n),
            np.ones(self.n),
            np.zeros(self.n),
        )

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        params = action_to_params(action)
        self.previous_state = self.state
        self.state = step_cpg(self.state, params, self.dt)

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        self.previous_action = action

        return obs, reward, terminated, truncated, info

    def _angular_velocity(self) -> float:
        if self.previous_state is None:
            return 0.0

        return self.state.phase[0] - self.previous_state.phase[0] / self.dt

    def _get_distance(self) -> float:
        raise NotImplementedError


def sdf_square(p, half_size):
    q = np.abs(p) - half_size
    return np.linalg.norm(np.maximum(q, 0)) + np.min(np.maximum(q[0], q[1]))


class SquareCPGEnv(CPGEnv):
    half_size = 1.0

    def _get_distance(self) -> float:
        x = self.state.amplitude[0] * np.cos(self.state.phase[0])
        y = self.state.amplitude[0] * np.sin(self.state.phase[0])
        return abs(sdf_square(np.array([x, y]), self.half_size))


gym.register(id="SquareCPGEnv-v0", entry_point=SquareCPGEnv)

if __name__ == "__main__":
    from visualization import animate_trajectory, plot_polar_trajectory, plot_trajectory

    env = gym.make("SquareCPGEnv-v0", n=2)
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
