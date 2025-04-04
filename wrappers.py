import dataclasses
from typing import Any, Callable, TypeAlias

from gymnax.environments import spaces
from gymnax.environments.environment import Environment, TEnvParams, TEnvState
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Key, PyTree


class GymnaxWrapper[TEnvState, TEnvParams]:
    def __init__(
        self,
        env: "GymnaxWrapper[TEnvState, TEnvParams] | Environment",
    ):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @property
    def unwrapped(
        self,
    ) -> Environment:
        if isinstance(self._env, GymnaxWrapper):
            return self._env.unwrapped
        else:
            return self._env


Env: TypeAlias = GymnaxWrapper[TEnvState, TEnvParams] | Environment


class TransformObservationWrapper(GymnaxWrapper[TEnvState, TEnvParams]):
    def __init__(
        self,
        env: Env[TEnvState, TEnvParams],
        observation_transform: Callable[
            [Any, TEnvState, TEnvParams], tuple[Array, TEnvState]
        ],
        observation_space_transform: Callable[[Any, TEnvParams], spaces.Space],
    ):
        super().__init__(env)
        self.observation_transform = observation_transform
        self.observation_space_transform = observation_space_transform

    def reset(self, key: Key, params: TEnvParams) -> tuple[Array, TEnvState]:
        observation, state = self._env.reset(key, params)
        return self.observation_transform(observation, state, params)

    def step(
        self, key: Key, state: TEnvState, action, params: TEnvParams
    ) -> tuple[Array, TEnvState, Float[Array, ""], Bool[Array, ""], dict[str, Any]]:
        observation, state, reward, done, info = self._env.step(
            key, state, action, params
        )

        observation, state = self.observation_transform(observation, state, params)

        return (
            observation,
            state,
            reward,
            done,
            info,
        )

    def observation_space(self, params: TEnvParams) -> spaces.Space:
        space = self._env.observation_space(params)
        return self.observation_space_transform(space, params)


class MaskObservationWrapper(TransformObservationWrapper[TEnvState, TEnvParams]):
    def __init__(self, env: Env[TEnvState, TEnvParams], mask: Bool[Array, "..."]):
        def mask_observation(
            observation: Array, state: TEnvState, params: TEnvParams
        ) -> tuple[Array, TEnvState]:
            observation_space = env.observation_space(params)

            if not isinstance(observation_space, spaces.Box):
                raise ValueError(
                    f"Only Box observation space supported for masking not {observation_space}"
                )

            return observation[mask], state

        def mask_observation_space(
            observation_space: spaces.Space, params: TEnvParams
        ) -> spaces.Box:
            if not isinstance(observation_space, spaces.Box):
                raise ValueError(
                    f"Only Box observation space supported for masking not {observation_space}"
                )

            low = jnp.broadcast_to(observation_space.low, observation_space.shape)[mask]
            high = jnp.broadcast_to(observation_space.high, observation_space.shape)[
                mask
            ]

            return spaces.Box(
                low=low,
                high=high,
                shape=mask.shape,
            )

        super().__init__(env, mask_observation, mask_observation_space)


class TransformRewardWrapper(GymnaxWrapper[TEnvState, TEnvParams]):
    def __init__(
        self,
        env: Env[TEnvState, TEnvParams],
        reward_transform: Callable[
            [Float[Array, ""], TEnvState, TEnvParams],
            tuple[Float[Array, ""], TEnvState],
        ],
    ):
        super().__init__(env)
        self.reward_transform = reward_transform

    def step(
        self, key: Key, state: TEnvState, action, params: TEnvParams
    ) -> tuple[Array, TEnvState, Float[Array, ""], Bool[Array, ""], dict[str, Any]]:
        observation, state, reward, done, info = self._env.step(
            key, state, action, params
        )

        reward, state = self.reward_transform(reward, state, params)

        return (
            jnp.asarray(observation),
            state,
            reward,
            done,
            info,
        )


class ClipRewardWrapper(TransformRewardWrapper[TEnvState, TEnvParams]):
    def __init__(
        self,
        env: Env[TEnvState, TEnvParams],
        min_reward: Float[ArrayLike, ""] = -10.0,
        max_reward: Float[ArrayLike, ""] = 10.0,
    ):
        def clip_reward(
            reward: Float[Array, ""], state: TEnvState, params: TEnvParams
        ) -> tuple[Float[Array, ""], TEnvState]:
            return jnp.clip(reward, min_reward, max_reward), state

        super().__init__(env, clip_reward)


class TransformActionWrapper(GymnaxWrapper[TEnvState, TEnvParams]):
    def __init__(
        self,
        env: Env[TEnvState, TEnvParams],
        action_transform: Callable[
            [Array, TEnvState, TEnvParams], tuple[Array, TEnvState]
        ],
        action_space_transform: Callable[[spaces.Space, TEnvParams], spaces.Space],
    ):
        super().__init__(env)
        self.action_transform = action_transform
        self.action_space_transform = action_space_transform

    def step(
        self, key: Key, state: TEnvState, action: Any, params: TEnvParams
    ) -> tuple[Array, TEnvState, Float[Array, ""], Bool[Array, ""], dict[str, Any]]:
        action, state = self.action_transform(action, state, params)

        observation, state, reward, done, info = self._env.step(
            key, state, action, params
        )

        return (
            jnp.asarray(observation),
            state,
            reward,
            done,
            info,
        )

    def action_space(self, params: TEnvParams) -> spaces.Space:
        return self.action_space_transform(self._env.action_space(params), params)


class ClipActionWrapper(TransformActionWrapper[TEnvState, TEnvParams]):
    def __init__(self, env: Env[TEnvState, TEnvParams]):
        def clip_action(
            action: Array, state: TEnvState, params: TEnvParams
        ) -> tuple[Array, TEnvState]:
            action_space = env.action_space(params)

            if not isinstance(action_space, spaces.Box):
                raise ValueError(
                    f"Only Box action space supported for clipping not {action_space}"
                )

            return jnp.clip(action, min=action_space.low, max=action_space.high), state

        super().__init__(env, clip_action, lambda x, _: x)


class RescaleAction(TransformActionWrapper[TEnvState, TEnvParams]):
    def __init__(
        self,
        env: Env[TEnvState, TEnvParams],
        min_action: Float[ArrayLike, ""] = jnp.array(-1),
        max_action: Float[ArrayLike, ""] = jnp.array(1),
    ):
        min_action = jnp.asarray(min_action)
        max_action = jnp.asarray(max_action)

        def rescale(
            arr: Array, state: TEnvState, params: TEnvParams
        ) -> tuple[Array, TEnvState]:
            action_space = env.action_space(params)

            # action_space = eqx.error_if(
            #     action_space,
            #     ~jnp.all(jnp.isfinite(action_space.low))
            #     | ~jnp.all(jnp.isfinite(action_space.high)),
            #     "Only finite spaces supported for rescaling.",
            # )

            grad = (action_space.high - action_space.low) / (max_action - min_action)
            intercept = action_space.low - grad * min_action
            return arr * grad + intercept, state

        def rescale_action_space(
            action_space: spaces.Space, params: TEnvParams
        ) -> spaces.Box:
            if not isinstance(action_space, spaces.Box):
                raise ValueError(
                    f"Only Box supported for rescaling not {action_space}."
                )

            low = jnp.broadcast_to(min_action, jnp.array(action_space.low).shape)
            high = jnp.broadcast_to(max_action, jnp.array(action_space.high).shape)
            return spaces.Box(low, high, action_space.shape)

        super().__init__(env, rescale, rescale_action_space)


class AddTimeWrapper(GymnaxWrapper[TEnvState, TEnvParams]):
    def __init__(
        self,
        env: Env,
        t0: Float[ArrayLike, ""] = jnp.array(0.0),
        dt: Float[ArrayLike, ""] = jnp.array(1.0),
    ):
        self._env = env

        self.env_has_time = None

        self.t0 = jnp.asarray(t0)
        self.dt = jnp.asarray(dt)

    def reset(self, key: Key, params: TEnvParams) -> tuple[PyTree[Array], TEnvState]:
        observation, state = self._env.reset(key, params)

        if hasattr(state, "time"):
            self.env_has_time = True
        else:
            state = dataclasses.replace(state, time=self.t0)

        observation = {
            "time": jnp.astype(state.time, jnp.float64),
            "observation": observation,
        }

        return observation, state

    def step(
        self, key: Key, state: TEnvState, action, params: TEnvParams
    ) -> tuple[
        PyTree[Array], TEnvState, Float[Array, ""], Bool[Array, ""], dict[str, Any]
    ]:
        observation, state, reward, done, info = self._env.step(
            key, state, action, params
        )

        if not self.env_has_time:
            state = dataclasses.replace(state, time=state.time + self.dt)

        observation = {
            "time": jnp.astype(state.time, jnp.float64),
            "observation": observation,
        }

        return (
            observation,
            state,
            reward,
            done,
            info,
        )

    def observation_space(self, params: TEnvParams) -> spaces.Dict:
        observation_space = self._env.observation_space(params)

        return spaces.Dict(
            {
                "time": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=()),
                "observation": observation_space,
            }
        )


class LogWrapper(GymnaxWrapper[TEnvState, TEnvParams]):
    pass
