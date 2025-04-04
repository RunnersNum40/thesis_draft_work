from typing import TypeVar
import dataclasses
import logging
from functools import partial

import chex
import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Key, PyTree

logger = logging.getLogger(__name__)

TObservation = TypeVar("TObservation", bound=PyTree[Array])
TAction = TypeVar("TAction", bound=PyTree[Array])


@chex.dataclass
class RolloutBuffer[TObservation, TAction]:
    observations: PyTree[TObservation]
    actions: PyTree[TAction]
    rewards: Float[Array, " *num_steps"]
    advantages: Float[Array, " *num_steps"]
    returns: Float[Array, " *num_steps"]
    terminations: Bool[Array, " *num_steps"]
    truncations: Bool[Array, " *num_steps"]
    log_probs: Float[Array, " *num_steps"]
    values: Float[Array, " *num_steps"]
    states: PyTree[eqx.nn.State]

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.rewards.shape[0],)


def compute_returns_and_advantages(
    buffer: RolloutBuffer[TObservation, TAction],
    last_value: Float[ArrayLike, ""],
    gamma: float,
    gae_lambda: float,
) -> RolloutBuffer[TObservation, TAction]:
    """Return a rollout buffer with returns and advantages added."""
    dones = jnp.logical_or(buffer.terminations, buffer.truncations).astype(jnp.float32)
    next_values = jnp.concatenate([buffer.values[1:], jnp.array([last_value])], axis=0)
    next_non_terminal = jnp.concatenate(
        [1.0 - dones[1:], jnp.array([1.0 - dones[-1]])], axis=0
    )

    deltas = buffer.rewards + gamma * next_values * next_non_terminal - buffer.values

    def scan_fn(
        carry: Float[Array, ""], xs: tuple[Float[Array, ""], Float[Array, ""]]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        delta, next_non_terminal = xs
        advantage = delta + gamma * gae_lambda * next_non_terminal * carry
        return advantage, advantage

    _, advantages_rev = jax.lax.scan(
        scan_fn, jnp.array(0.0), (jnp.flip(deltas), jnp.flip(next_non_terminal))
    )
    advantages = jnp.flip(advantages_rev)
    returns = advantages + buffer.values

    return dataclasses.replace(buffer, advantages=advantages, returns=returns)


def batches(
    buffer: RolloutBuffer[TObservation, TAction],
    batch_size: int,
    key: Key | None = None,
) -> RolloutBuffer[TObservation, TAction]:
    """Takes a rollout buffer of a single rollout and returns a rollout buffer of batches steps"""
    if key is None:
        indices = jnp.arange(buffer.shape[0])
    else:
        indices = jr.permutation(key, buffer.shape[0])

    if buffer.shape[0] % batch_size != 0:
        logger.warning(
            f"Buffer size {buffer.shape[0]} not compatible with batch size {batch_size}"
        )
        indices = indices[: buffer.shape[0] - buffer.shape[0] % batch_size]

    indices = indices.reshape(-1, batch_size)

    return jax.tree.map(partial(jnp.take, indices=indices, axis=0), buffer)
