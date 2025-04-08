from abc import abstractmethod
from typing import Callable, TypeVar

import equinox as eqx
import gymnax as gym
from distreqx import distributions
from gymnax.environments import spaces
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int, Key, PyTree

from stateful_ncde import NeuralCDE
from wrappers import Env

TFeatures = TypeVar("TFeatures", bound=PyTree[Array])
TAction = TypeVar("TAction", bound=PyTree[Array])
TObservation = TypeVar("TObservation", bound=PyTree[Array])
TActionSpace = TypeVar("TActionSpace", bound=spaces.Space)
TObservationSpace = TypeVar("TObservationSpace", bound=spaces.Space)


class AbstractActorCriticPolicy[
    TFeatures, TObservation, TAction, TActionSpace, TObservationSpace
](
    eqx.Module,
    strict=True,
):
    """Abstract base policy class for actor-critic algorithms.

    Has both a value and action prediction.
    """

    observation_space: eqx.AbstractVar[TObservationSpace]
    action_space: eqx.AbstractVar[TActionSpace]

    @abstractmethod
    def extract_features(
        self, observation: TObservation, state: eqx.nn.State
    ) -> tuple[TFeatures, eqx.nn.State]:
        """Compute a set of features from the observation."""
        raise NotImplementedError

    @abstractmethod
    def value_from_features(
        self, features: TFeatures, state: eqx.nn.State
    ) -> tuple[Array, eqx.nn.State]:
        raise NotImplementedError

    @abstractmethod
    def action_dist_from_features(
        self, features: TFeatures, state
    ) -> tuple[distributions.AbstractDistribution, eqx.nn.State]:
        raise NotImplementedError

    def __call__(
        self, observation: TObservation, state: eqx.nn.State, *, key: Key | None = None
    ) -> tuple[TAction, Float[Array, ""], Float[Array, ""], eqx.nn.State]:
        """Get a new action, value, and log probability of the action.

        If no key is passed the action will be generated deterministically.
        """
        features, state = self.extract_features(observation, state)
        value, state = self.value_from_features(features, state)
        action_dist, state = self.action_dist_from_features(features, state)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)
        log_prob = action_dist.log_prob(action)

        return action, value, log_prob.squeeze(), state

    def predict_action(
        self, observation: TObservation, state: eqx.nn.State, *, key: Key | None = None
    ) -> tuple[TAction, eqx.nn.State]:
        """Get an action."""
        features, state = self.extract_features(observation, state)
        action_dist, state = self.action_dist_from_features(features, state)

        if key is None:
            action = action_dist.mode()
        else:
            action = action_dist.sample(key)

        return action, state

    def predict_value(
        self, observation: TObservation, state: eqx.nn.State
    ) -> tuple[Float[Array, ""], eqx.nn.State]:
        features, state = self.extract_features(observation, state)
        value, state = self.value_from_features(features, state)
        return value, state

    def evaluate_action(
        self, observation: TObservation, action: TAction, state: eqx.nn.State
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], eqx.nn.State]:
        features, state = self.extract_features(observation, state)
        value, state = self.value_from_features(features, state)
        action_dist, state = self.action_dist_from_features(features, state)
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return value, log_prob.squeeze(), entropy.squeeze(), state

    @abstractmethod
    def reset(
        self, env_state: gym.EnvState, env_params: gym.EnvParams, state: eqx.nn.State
    ) -> eqx.nn.State:
        raise NotImplementedError

    @abstractmethod
    def get_hidden_state(self, state: eqx.nn.State) -> Float[Array, " state_size"]:
        raise NotImplementedError


def space_size(space: spaces.Space) -> Int:
    """Get the size of a space."""
    if isinstance(space, spaces.Discrete):
        return int(jnp.array(space.n))

    if isinstance(space, spaces.Box):
        return int(jnp.prod(jnp.asarray(space.shape)))

    if isinstance(space, spaces.Tuple):
        return int(jnp.sum(jnp.array([space_size(s) for s in space.spaces])))

    if isinstance(space, spaces.Dict):
        return int(jnp.sum(jnp.array([space_size(s) for s in space.spaces.values()])))

    raise ValueError(f"Cannot get size of space {space}.")


def flatten_sample(
    sample: PyTree[Float[Array, "..."]], space: spaces.Space
) -> Float[Array, " n"]:
    if isinstance(space, spaces.Discrete):
        return jnp.expand_dims(jnp.array(sample), axis=-1)

    if isinstance(space, spaces.Box):
        return jnp.ravel(sample)

    if isinstance(space, spaces.Tuple):
        return jnp.concatenate(
            [flatten_sample(s, space.spaces[i]) for i, s in enumerate(sample)]
        )

    if isinstance(space, spaces.Dict):
        if "time" in sample:
            time = jnp.expand_dims(jnp.array(sample["time"]), axis=-1) / 1000.0
            return jnp.concatenate(
                [
                    time,
                    flatten_sample(sample["observation"], space.spaces["observation"]),
                ]
            )
        else:
            return jnp.concatenate(
                [flatten_sample(sample[k], space.spaces[k]) for k in sample.keys()]
            )

    raise ValueError(f"Cannot flatten {space}.")


class MLPActorMLPCriticPolicy(
    AbstractActorCriticPolicy[
        Float[Array, " n"],
        Float[Array, "..."],
        PyTree[Array],
        spaces.Box | spaces.Discrete,
        spaces.Space,
    ],
    strict=True,
):
    observation_space: spaces.Space
    action_space: spaces.Box | spaces.Discrete

    actor: eqx.nn.MLP
    action_logstd: Float[Array, " ..."] | None
    critic: eqx.nn.MLP

    state_index: eqx.nn.StateIndex[Int[Array, ""]]

    def __init__(
        self,
        env: Env,
        env_params: gym.EnvParams,
        width_size: int,
        depth: int,
        *,
        actor_width_size: int | None = None,
        actor_depth: int | None = None,
        actor_activation: Callable[[Array], Array] = jnn.relu,
        actor_final_activation: Callable[[Array], Array] = lambda x: x,
        critic_width_size: int | None = None,
        critic_depth: int | None = None,
        critic_activation: Callable[[Array], Array] = jnn.relu,
        critic_final_activation: Callable[[Array], Array] = lambda x: x,
        key: Key,
    ):
        self.observation_space = env.observation_space(env_params)
        self.action_space = env.action_space(env_params)

        actor_key, critic_key = jr.split(key, 2)

        self.actor = eqx.nn.MLP(
            in_size=space_size(self.observation_space),
            out_size=space_size(self.action_space),
            width_size=actor_width_size if actor_width_size else width_size,
            depth=actor_depth if actor_depth else depth,
            activation=actor_activation,
            final_activation=actor_final_activation,
            key=actor_key,
        )

        if isinstance(self.action_space, spaces.Discrete):
            self.action_logstd = None
        elif isinstance(self.action_space, spaces.Box):
            self.action_logstd = jnp.zeros(space_size(self.action_space))
        else:
            raise ValueError(
                f"Cannot create actor for action space {self.action_space}. Only Discrete and Box supported."
            )

        self.critic = eqx.nn.MLP(
            in_size=space_size(self.observation_space),
            out_size="scalar",
            width_size=critic_width_size if critic_width_size else width_size,
            depth=critic_depth if critic_depth else depth,
            activation=critic_activation,
            final_activation=critic_final_activation,
            key=critic_key,
        )

        self.state_index = eqx.nn.StateIndex(jnp.array(0))

    def extract_features(
        self, observation: Array, state: eqx.nn.State
    ) -> tuple[Float[Array, " n"], eqx.nn.State]:
        """Extract features from the observation."""
        features = flatten_sample(observation, self.observation_space)
        return features, state

    def value_from_features(
        self, features: Float[Array, " n"], state: eqx.nn.State
    ) -> tuple[Float[Array, ""], eqx.nn.State]:
        """Get the value from the features."""
        value = self.critic(features)
        return value, state

    def action_dist_from_features(
        self, features: Float[Array, " n"], state: eqx.nn.State
    ) -> tuple[distributions.AbstractDistribution, eqx.nn.State]:
        """Get the action distribution from the features.

        Returns a Categorical distribution for discrete actions and a Normal
        distribution for continuous actions.
        """
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.actor(features)
            action_dist = distributions.Categorical(logits=logits)
        elif isinstance(self.action_space, spaces.Box):
            assert self.action_logstd is not None
            mean = self.actor(features)
            std = jnp.exp(self.action_logstd)
            action_dist = distributions.Normal(loc=mean, scale=std)
        else:
            raise ValueError(f"Action space {self.action_space} is not supported.")

        return action_dist, state

    def reset(
        self, env_state: gym.EnvState, env_params: gym.EnvParams, state: eqx.nn.State
    ) -> eqx.nn.State:
        """Reset the policy state."""
        return state

    def get_hidden_state(self, state: eqx.nn.State) -> Float[Array, " state_size"]:
        return state.get(self.state_index)


class SharedCDEPolicy(
    AbstractActorCriticPolicy[
        Float[Array, " n"],
        Float[Array, "..."],
        PyTree[Array],
        spaces.Box | spaces.Discrete,
        spaces.Dict,
    ],
    strict=True,
):
    observation_space: spaces.Dict
    action_space: spaces.Box | spaces.Discrete

    ncde: NeuralCDE

    actor: eqx.nn.MLP
    action_logstd: Float[Array, " ..."] | None
    critic: eqx.nn.MLP

    def __init__(
        self,
        env: Env,
        env_params: gym.EnvParams,
        width_size: int,
        depth: int,
        state_size: int,
        num_features: int,
        max_steps: int,
        *,
        initial_width_size: int | None = None,
        initial_depth: int | None = None,
        initial_activation: Callable[[Array], Array] = jnn.relu,
        inital_final_activation: Callable[[Array], Array] = lambda x: x,
        field_width_size: int | None = None,
        field_depth: int | None = None,
        field_activation: Callable[[Array], Array] = jnn.relu,
        field_final_activation: Callable[[Array], Array] = jnn.tanh,
        output_width_size: int | None = None,
        output_depth: int | None = None,
        output_activation: Callable[[Array], Array] = jnn.relu,
        output_final_activation: Callable[[Array], Array] = lambda x: x,
        actor_width_size: int | None = None,
        actor_depth: int | None = None,
        actor_activation: Callable[[Array], Array] = jnn.relu,
        actor_final_activation: Callable[[Array], Array] = lambda x: x,
        critic_width_size: int | None = None,
        critic_depth: int | None = None,
        critic_activation: Callable[[Array], Array] = jnn.relu,
        critic_final_activation: Callable[[Array], Array] = lambda x: x,
        key: Key,
    ):
        self.observation_space = env.observation_space(env_params)
        self.action_space = env.action_space(env_params)

        assert isinstance(self.observation_space, spaces.Dict)
        assert "time" in self.observation_space.spaces
        assert "observation" in self.observation_space.spaces
        assert len(self.observation_space.spaces) == 2

        ncde_key, actor_key, critic_key = jr.split(key, 3)

        self.ncde = NeuralCDE(
            input_size=space_size(self.observation_space.spaces["observation"]),
            state_size=state_size,
            output_size=num_features,
            width_size=width_size,
            depth=depth,
            max_steps=max_steps,
            initial_width_size=initial_width_size,
            initial_depth=initial_depth,
            field_width_size=field_width_size,
            field_depth=field_depth,
            output_width_size=output_width_size,
            output_depth=output_depth,
            field_activation=field_activation,
            field_final_activation=field_final_activation,
            initial_activation=initial_activation,
            initial_final_activation=inital_final_activation,
            output_activation=output_activation,
            output_final_activation=output_final_activation,
            key=ncde_key,
        )

        self.actor = eqx.nn.MLP(
            in_size=num_features,
            out_size=space_size(self.action_space),
            width_size=actor_width_size if actor_width_size else width_size,
            depth=actor_depth if actor_depth else depth,
            activation=actor_activation,
            final_activation=actor_final_activation,
            key=actor_key,
        )

        if isinstance(self.action_space, spaces.Discrete):
            self.action_logstd = None
        elif isinstance(self.action_space, spaces.Box):
            self.action_logstd = jnp.zeros(space_size(self.action_space))
        else:
            raise ValueError(
                f"Cannot create actor for action space {self.action_space}. Only Discrete and Box supported."
            )

        self.critic = eqx.nn.MLP(
            in_size=num_features,
            out_size="scalar",
            width_size=critic_width_size if critic_width_size else width_size,
            depth=critic_depth if critic_depth else depth,
            activation=critic_activation,
            final_activation=critic_final_activation,
            key=critic_key,
        )

    def extract_features(
        self, observation: PyTree[Array], state: eqx.nn.State
    ) -> tuple[Float[Array, " n"], eqx.nn.State]:
        """Extract features from the observation."""
        time = observation["time"]
        flattened = flatten_sample(
            observation["observation"], self.observation_space.spaces["observation"]
        )

        features, state = self.ncde(time, flattened, state)
        return features, state

    def value_from_features(
        self, features: Float[Array, " n"], state: eqx.nn.State
    ) -> tuple[Float[Array, ""], eqx.nn.State]:
        """Get the value from the features."""
        value = self.critic(features)
        return value, state

    def action_dist_from_features(
        self, features: Float[Array, " n"], state: eqx.nn.State
    ) -> tuple[distributions.AbstractDistribution, eqx.nn.State]:
        """Get the action distribution from the features.

        Returns a Categorical distribution for discrete actions and a Normal
        distribution for continuous actions.
        """
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.actor(features)
            action_dist = distributions.Categorical(logits=logits)
        elif isinstance(self.action_space, spaces.Box):
            assert self.action_logstd is not None
            mean = self.actor(features)
            std = jnp.exp(self.action_logstd)
            action_dist = distributions.Normal(loc=mean, scale=std)
        else:
            raise ValueError(f"Action space {self.action_space} is not supported.")

        return action_dist, state

    def reset(
        self, env_state: gym.EnvState, env_params: gym.EnvParams, state: eqx.nn.State
    ) -> eqx.nn.State:
        """Reset the policy state."""
        return self.ncde.empty_state(state)

    def get_hidden_state(self, state: eqx.nn.State) -> Float[Array, " state_size"]:
        return self.ncde.z1(state)


class CDEActorCDECriticPolicy(
    AbstractActorCriticPolicy[
        tuple[Float[Array, ""], Float[Array, " n"]],
        Float[Array, "..."],
        PyTree[Array],
        spaces.Box | spaces.Discrete,
        spaces.Dict,
    ],
    strict=True,
):
    observation_space: spaces.Dict
    action_space: spaces.Box | spaces.Discrete

    actor: NeuralCDE
    action_logstd: Float[Array, " ..."] | None
    critic: NeuralCDE

    def __init__(
        self,
        env: Env,
        env_params: gym.EnvParams,
        width_size: int,
        depth: int,
        state_size: int,
        max_steps: int,
        *,
        initial_width_size: int | None = None,
        initial_depth: int | None = None,
        initial_activation: Callable[[Array], Array] = jnn.relu,
        inital_final_activation: Callable[[Array], Array] = lambda x: x,
        field_width_size: int | None = None,
        field_depth: int | None = None,
        field_activation: Callable[[Array], Array] = jnn.relu,
        field_final_activation: Callable[[Array], Array] = jnn.tanh,
        output_width_size: int | None = None,
        output_depth: int | None = None,
        output_activation: Callable[[Array], Array] = jnn.relu,
        output_final_activation: Callable[[Array], Array] = lambda x: x,
        key: Key,
    ):
        self.observation_space = env.observation_space(env_params)
        self.action_space = env.action_space(env_params)

        assert isinstance(self.observation_space, spaces.Dict)
        assert "time" in self.observation_space.spaces
        assert "observation" in self.observation_space.spaces
        assert len(self.observation_space.spaces) == 2

        actor_key, critic_key = jr.split(key, 2)

        self.actor = NeuralCDE(
            input_size=space_size(self.observation_space.spaces["observation"]),
            state_size=state_size,
            output_size=space_size(self.action_space),
            width_size=width_size,
            depth=depth,
            max_steps=max_steps,
            initial_width_size=initial_width_size,
            initial_depth=initial_depth,
            field_width_size=field_width_size,
            field_depth=field_depth,
            output_width_size=output_width_size,
            output_depth=output_depth,
            field_activation=field_activation,
            field_final_activation=field_final_activation,
            initial_activation=initial_activation,
            initial_final_activation=inital_final_activation,
            output_activation=output_activation,
            output_final_activation=output_final_activation,
            key=actor_key,
        )

        if isinstance(self.action_space, spaces.Discrete):
            self.action_logstd = None
        elif isinstance(self.action_space, spaces.Box):
            self.action_logstd = jnp.zeros(space_size(self.action_space))
        else:
            raise ValueError(
                f"Cannot create actor for action space {self.action_space}. Only Discrete and Box supported."
            )

        self.critic = NeuralCDE(
            input_size=space_size(self.observation_space.spaces["observation"]),
            state_size=state_size,
            output_size="scalar",
            width_size=width_size,
            depth=depth,
            max_steps=max_steps,
            initial_width_size=initial_width_size,
            initial_depth=initial_depth,
            field_width_size=field_width_size,
            field_depth=field_depth,
            output_width_size=output_width_size,
            output_depth=output_depth,
            field_activation=field_activation,
            field_final_activation=field_final_activation,
            initial_activation=initial_activation,
            initial_final_activation=inital_final_activation,
            output_activation=output_activation,
            output_final_activation=output_final_activation,
            key=critic_key,
        )

    def extract_features(
        self, observation: PyTree[Array], state: eqx.nn.State
    ) -> tuple[tuple[Float[Array, ""], Float[Array, " n"]], eqx.nn.State]:
        """Extract features from the observation."""
        time = observation["time"]
        flattened = flatten_sample(
            observation["observation"], self.observation_space.spaces["observation"]
        )

        return (time, flattened), state

    def value_from_features(
        self, features: tuple[Float[Array, ""], Float[Array, " n"]], state: eqx.nn.State
    ) -> tuple[Float[Array, ""], eqx.nn.State]:
        """Get the value from the features."""
        value, state = self.critic(*features, state)
        return value, state

    def action_dist_from_features(
        self, features: tuple[Float[Array, ""], Float[Array, " n"]], state: eqx.nn.State
    ) -> tuple[distributions.AbstractDistribution, eqx.nn.State]:
        """Get the action distribution from the features.

        Returns a Categorical distribution for discrete actions and a Normal
        distribution for continuous actions.
        """
        if isinstance(self.action_space, spaces.Discrete):
            logits, state = self.actor(*features, state)
            action_dist = distributions.Categorical(logits=logits)
        elif isinstance(self.action_space, spaces.Box):
            assert self.action_logstd is not None
            mean, state = self.actor(*features, state)
            std = jnp.exp(self.action_logstd)
            action_dist = distributions.Normal(loc=mean, scale=std)
        else:
            raise ValueError(f"Action space {self.action_space} is not supported.")

        return action_dist, state

    def reset(
        self, env_state: gym.EnvState, env_params: gym.EnvParams, state: eqx.nn.State
    ) -> eqx.nn.State:
        """Reset the policy state."""
        state = self.actor.empty_state(state)
        state = self.critic.empty_state(state)
        return state

    def get_hidden_state(self, state: eqx.nn.State) -> Float[Array, " state_size"]:
        return jnp.concatenate([self.actor.z1(state), self.critic.z1(state)])


class CDEActorMLPCriticPolicy(
    AbstractActorCriticPolicy[
        tuple[Float[Array, ""], Float[Array, " n"]],
        Float[Array, "..."],
        PyTree[Array],
        spaces.Box | spaces.Discrete,
        spaces.Dict,
    ],
    strict=True,
):
    observation_space: spaces.Dict
    action_space: spaces.Box | spaces.Discrete

    actor: NeuralCDE
    action_logstd: Float[Array, " ..."] | None
    critic: eqx.nn.MLP

    def __init__(
        self,
        env: Env,
        env_params: gym.EnvParams,
        width_size: int,
        depth: int,
        state_size: int,
        max_steps: int,
        *,
        initial_width_size: int | None = None,
        initial_depth: int | None = None,
        initial_activation: Callable[[Array], Array] = jnn.relu,
        inital_final_activation: Callable[[Array], Array] = lambda x: x,
        field_width_size: int | None = None,
        field_depth: int | None = None,
        field_activation: Callable[[Array], Array] = jnn.relu,
        field_final_activation: Callable[[Array], Array] = jnn.tanh,
        output_width_size: int | None = None,
        output_depth: int | None = None,
        output_activation: Callable[[Array], Array] = jnn.relu,
        output_final_activation: Callable[[Array], Array] = lambda x: x,
        critic_width_size: int | None = None,
        critic_depth: int | None = None,
        critic_activation: Callable[[Array], Array] = jnn.relu,
        critic_final_activation: Callable[[Array], Array] = lambda x: x,
        key: Key,
    ):
        self.observation_space = env.observation_space(env_params)
        self.action_space = env.action_space(env_params)

        assert isinstance(self.observation_space, spaces.Dict)
        assert "time" in self.observation_space.spaces
        assert "observation" in self.observation_space.spaces
        assert len(self.observation_space.spaces) == 2

        actor_key, critic_key = jr.split(key, 2)

        self.actor = NeuralCDE(
            input_size=space_size(self.observation_space.spaces["observation"]),
            state_size=state_size,
            output_size=space_size(self.action_space),
            width_size=width_size,
            depth=depth,
            max_steps=max_steps,
            initial_width_size=initial_width_size,
            initial_depth=initial_depth,
            field_width_size=field_width_size,
            field_depth=field_depth,
            output_width_size=output_width_size,
            output_depth=output_depth,
            field_activation=field_activation,
            field_final_activation=field_final_activation,
            initial_activation=initial_activation,
            initial_final_activation=inital_final_activation,
            output_activation=output_activation,
            output_final_activation=output_final_activation,
            key=actor_key,
        )

        if isinstance(self.action_space, spaces.Discrete):
            self.action_logstd = None
        elif isinstance(self.action_space, spaces.Box):
            self.action_logstd = jnp.zeros(space_size(self.action_space))
        else:
            raise ValueError(
                f"Cannot create actor for action space {self.action_space}. Only Discrete and Box supported."
            )

        self.critic = eqx.nn.MLP(
            in_size=space_size(self.observation_space.spaces["observation"]),
            out_size="scalar",
            width_size=critic_width_size if critic_width_size else width_size,
            depth=critic_depth if critic_depth else depth,
            activation=critic_activation,
            final_activation=critic_final_activation,
            key=critic_key,
        )

    def extract_features(
        self, observation: PyTree[Array], state: eqx.nn.State
    ) -> tuple[tuple[Float[Array, ""], Float[Array, " n"]], eqx.nn.State]:
        """Extract features from the observation."""
        time = observation["time"] * 1e-3
        flattened = flatten_sample(
            observation["observation"], self.observation_space.spaces["observation"]
        )

        return (time, flattened), state

    def value_from_features(
        self, features: tuple[Float[Array, ""], Float[Array, " n"]], state: eqx.nn.State
    ) -> tuple[Float[Array, ""], eqx.nn.State]:
        """Get the value from the features."""
        value = self.critic(features[1])
        return value, state

    def action_dist_from_features(
        self, features: tuple[Float[Array, ""], Float[Array, " n"]], state: eqx.nn.State
    ) -> tuple[distributions.AbstractDistribution, eqx.nn.State]:
        """Get the action distribution from the features.

        Returns a Categorical distribution for discrete actions and a Normal
        distribution for continuous actions.
        """
        if isinstance(self.action_space, spaces.Discrete):
            logits, state = self.actor(*features, state)
            action_dist = distributions.Categorical(logits=logits)
        elif isinstance(self.action_space, spaces.Box):
            assert self.action_logstd is not None
            mean, state = self.actor(*features, state)
            std = jnp.exp(self.action_logstd)
            action_dist = distributions.Normal(loc=mean, scale=std)
        else:
            raise ValueError(f"Action space {self.action_space} is not supported.")

        return action_dist, state

    def reset(
        self, env_state: gym.EnvState, env_params: gym.EnvParams, state: eqx.nn.State
    ) -> eqx.nn.State:
        """Reset the policy state."""
        return self.actor.empty_state(state)

    def get_hidden_state(self, state: eqx.nn.State) -> Float[Array, " state_size"]:
        return self.actor.z1(state)
