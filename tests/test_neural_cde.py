import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax import random as jr

from cde_actor import (
    CDEAgent,
    EpisodesRollout,
    PPOArguments,
    collect_ppo_rollout,
    compute_gae_episode,
    env_step,
    get_batch_indices,
    reset,
    split_into_episodes,
    train,
)


# --- Dummy Environment and Fixtures ---


class DummyEnv:
    """A minimal dummy environment for testing."""

    def reset(self, key, params):
        # Return a fixed observation.
        obs = jnp.array([0.5, 0.5, 0.5])
        return obs, {"state": 0}

    def step(self, key, state, action, params):
        # Always return the same observation, reward, and not done.
        obs = jnp.array([0.5, 0.5, 0.5])
        reward = 1.0
        done = False
        info = {}
        return obs, state, reward, done, info

    def observation_space(self, params):
        class DummySpace:
            shape = (3,)

        return DummySpace()

    def action_space(self, params):
        class DummySpace:
            shape = (1,)
            low = jnp.array([-1.0])
            high = jnp.array([1.0])

        return DummySpace()


@pytest.fixture
def agent_actual():
    """Fixture that creates a CDEAgent using the DummyEnv."""
    key = jr.key(0)
    env = DummyEnv()
    params = {}
    agent = CDEAgent(
        env=env,
        env_params=params,
        hidden_size=4,
        processed_size=3,
        width_size=16,
        depth=2,
        key=key,
    )
    return agent, env, params, key


@pytest.fixture
def ppo_args():
    """Fixture for PPOArguments."""
    return PPOArguments(
        num_steps=10,
        gamma=0.99,
        gae_lambda=0.95,
        num_epochs=1,
        normalize_advantage=True,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        max_gradient_norm=0.5,
        target_kl=None,
        minibatch_size=2,
        num_iterations=1,
        learning_rate=1e-3,
        anneal_learning_rate=False,
    )


# --- Sanity Checks ---


def test_initial_state(agent_actual):
    """Test that the agent's initial state has the expected shape."""
    agent, _, _, key = agent_actual
    t0 = jnp.array(0.0)
    x0 = jnp.array([0.5, 0.5, 0.5])
    state = agent.initial_state(t0, x0, key=key)
    assert state.shape == (agent.state_size,)


def test_get_value(agent_actual):
    """Test that get_value returns valid outputs without NaNs."""
    agent, _, _, key = agent_actual
    ts = jnp.linspace(0.0, 1.0, 10)
    xs = jnp.tile(jnp.array([0.5, 0.5, 0.5]), (10, 1))
    z0 = jnp.zeros(agent.state_size)
    value = agent.get_value(ts, xs, z0, key=key, evolving_out=False)
    assert not jnp.isnan(value).any()


def test_get_action_and_value(agent_actual):
    """Test that get_action_and_value returns a tuple of the expected length."""
    agent, _, _, key = agent_actual
    ts = jnp.linspace(0.0, 1.0, 10)
    xs = jnp.tile(jnp.array([0.5, 0.5, 0.5]), (10, 1))
    z0 = jnp.zeros(agent.state_size)
    a_fixed = jnp.array([[0.1]] * 10)
    outputs = agent.get_action_and_value(
        ts, xs, z0, a1=a_fixed, key=key, evolving_out=True
    )
    assert len(outputs) == 5


def test_reset(agent_actual, ppo_args):
    """Test that reset initializes the episode state with valid observations and times."""
    agent, env, params, key = agent_actual
    ep_state = reset(env, params, ppo_args, key=key)
    assert not jnp.isnan(ep_state.observations[0]).all()
    assert not jnp.isnan(ep_state.times[0])


def test_env_step(agent_actual, ppo_args):
    """Test that env_step updates the episode state with a valid new observation."""
    agent, env, params, key = agent_actual
    ep_state = reset(env, params, ppo_args, key=key)
    new_state, buffer = env_step(env, params, agent, ep_state, ppo_args, key=key)
    assert not jnp.isnan(new_state.observations[new_state.step]).all()


def test_compute_gae_episode():
    """Test that compute_gae_episode produces outputs without NaNs."""
    rewards = jnp.array([1.0] * 5)
    values = jnp.array([0.5] * 5)
    terminations = jnp.array([False] * 5)
    truncations = jnp.array([False] * 5)
    args = PPOArguments(
        num_steps=5,
        gamma=0.99,
        gae_lambda=0.95,
        num_epochs=1,
        normalize_advantage=True,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        max_gradient_norm=0.5,
        target_kl=None,
        minibatch_size=2,
        num_iterations=1,
        learning_rate=1e-3,
        anneal_learning_rate=False,
    )
    returns, advantages = compute_gae_episode(
        rewards, values, terminations, truncations, args
    )
    assert not jnp.isnan(returns).any()
    assert not jnp.isnan(advantages).any()


def test_get_batch_indices():
    """Test that get_batch_indices returns indices with the correct shape."""
    batch_size = 3
    dataset_size = 10
    key = jr.key(0)
    indices = get_batch_indices(batch_size, dataset_size, key=key)
    assert indices.shape == (dataset_size // batch_size, batch_size)


def test_non_monotonic_ts_get_value(agent_actual):
    """Test that get_value raises an exception when time array is non-monotonic."""
    agent, _, _, key = agent_actual
    ts = jnp.array([0.0, 0.5, 0.4, 0.8, 1.0])
    xs = jnp.tile(jnp.array([0.5, 0.5, 0.5]), (5, 1))
    z0 = jnp.zeros(agent.state_size)
    with pytest.raises(Exception, match="monotonically strictly increasing"):
        agent.get_value(ts, xs, z0, key=key, evolving_out=False)


def test_train(agent_actual, ppo_args):
    """Test that a full training loop updates at least one parameter of the agent."""
    agent, env, params, key = agent_actual
    new_agent = train(env, params, agent, ppo_args, key=key)
    assert isinstance(new_agent, type(agent))
    orig_params = eqx.filter(agent, eqx.is_array)
    new_params = eqx.filter(new_agent, eqx.is_array)
    orig_leaves = jax.tree.leaves(orig_params)
    new_leaves = jax.tree.leaves(new_params)
    changes = [not jnp.allclose(op, np) for op, np in zip(orig_leaves, new_leaves)]
    assert any(changes), "Expected at least one parameter to change after training."


# --- Monotonicity Tests for Episode Times ---


def test_env_step_monotonic_times(agent_actual, ppo_args):
    """Test that repeated calls to env_step produce strictly increasing times."""
    agent, env, params, key = agent_actual
    ep_state = reset(env, params, ppo_args, key=key)
    times = []
    for _ in range(5):
        ep_state, _ = env_step(env, params, agent, ep_state, ppo_args, key=key)
        curr_time = ep_state.times[ep_state.step]
        if not jnp.isnan(curr_time):
            times.append(curr_time)
    assert all(
        t1 < t2 for t1, t2 in zip(times, times[1:])
    ), "Times should be strictly increasing."


def test_collect_ppo_rollout_monotonic(agent_actual, ppo_args):
    """Test that collect_ppo_rollout produces episodes with strictly increasing times."""
    agent, env, params, key = agent_actual
    ep_state = reset(env, params, ppo_args, key=key)
    _, _, rollout = collect_ppo_rollout(
        env, params, agent, ep_state, None, ppo_args, key=key
    )
    for episode in rollout.times:
        valid_times = episode[~jnp.isnan(episode)]
        if valid_times.size > 1:
            assert jnp.all(
                valid_times[:-1] < valid_times[1:]
            ), "Episode times must be strictly increasing."


def test_split_into_episodes_monotonic(ppo_args):
    """Test that split_into_episodes correctly segments a concatenated rollout with monotonic times."""
    num_steps = ppo_args.num_steps  # e.g., 10
    episode_length = num_steps // 2  # simulate two episodes

    # Create concatenated rollout fields.
    times = jnp.concatenate(
        [jnp.linspace(0, 1, episode_length), jnp.linspace(0, 1, episode_length)]
    )
    observations = jnp.tile(jnp.array([0.5, 0.5, 0.5]), (num_steps, 1))
    actions = jnp.concatenate(
        [jnp.full((episode_length, 1), 0.1), jnp.full((episode_length, 1), 0.2)]
    )
    log_probs = jnp.concatenate(
        [jnp.linspace(0, -1, episode_length), jnp.linspace(-1, -2, episode_length)]
    )
    entropies = jnp.concatenate(
        [jnp.linspace(1, 0.5, episode_length), jnp.linspace(0.5, 0.0, episode_length)]
    )
    values = jnp.concatenate(
        [jnp.linspace(0.5, 1, episode_length), jnp.linspace(1, 1.5, episode_length)]
    )
    rewards = jnp.concatenate([jnp.ones(episode_length), jnp.ones(episode_length)])
    terminations = jnp.concatenate(
        [
            jnp.array([False] * (episode_length - 1) + [True]),
            jnp.array([False] * (episode_length - 1) + [True]),
        ]
    )
    truncations = jnp.full((num_steps,), False)
    advantages = jnp.concatenate(
        [jnp.linspace(0, 1, episode_length), jnp.linspace(1, 2, episode_length)]
    )
    returns = jnp.concatenate(
        [jnp.linspace(0, 1, episode_length), jnp.linspace(1, 2, episode_length)]
    )
    initial_agent_state = jnp.tile(jnp.array([0.0, 0.0]), (num_steps, 1))

    dummy_rollout = EpisodesRollout(
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        entropies=entropies,
        values=values,
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        advantages=advantages,
        returns=returns,
        times=times,
        initial_agent_state=initial_agent_state,
    )
    episodes = split_into_episodes(dummy_rollout, ppo_args)
    for ep in episodes.times:
        valid_times = ep[~jnp.isnan(ep)]
        if valid_times.size > 1:
            assert jnp.all(
                valid_times[:-1] < valid_times[1:]
            ), "Split episodes should have strictly increasing times."
