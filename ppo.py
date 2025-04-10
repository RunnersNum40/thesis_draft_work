from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

import equinox as eqx
import gymnax as gym
from gymnax import wrappers
import jax
import numpy as np
import optax
from gymnax.environments import spaces
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key, PyTree
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.rich import tqdm

from buffers import RolloutBuffer, batches, compute_returns_and_advantages
from policies import AbstractActorCriticPolicy
from wrappers import Env


def remove_weak_types(pytree: PyTree) -> PyTree:
    """Remove weak types from a PyTree."""
    return jax.tree.map(lambda x: jnp.asarray(x, jnp.asarray(x).dtype), pytree)


def fix_env_state(env_state: gym.EnvState) -> gym.EnvState:
    """Ensure consistent typing of the environment state."""
    env_state = remove_weak_types(env_state)
    env_state = eqx.tree_at(
        lambda s: s.episode_returns,
        env_state,
        replace_fn=lambda x: jnp.astype(x, jnp.float64),
    )
    env_state = eqx.tree_at(
        lambda s: s.returned_episode_returns,
        env_state,
        replace_fn=lambda x: jnp.astype(x, jnp.float64),
    )

    return env_state


TArg = ParamSpec("TArg")


def debug_wrapper(
    func: Callable[TArg, Any], ordered: bool = False
) -> Callable[TArg, None]:
    """Return a new version of a function that works inside a JIT region."""

    def cast_to_numpy(*args: TArg.args, **kwargs: TArg.kwargs):
        """Convert all JAX arrays to NumPy arrays and call the function."""
        (args, kwargs) = jax.tree.map(
            lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x,
            (args, kwargs),
        )
        func(*args, **kwargs)

    @wraps(func)
    def wrapped(*args, **kwargs):
        """Call the function with a callback."""
        jax.debug.callback(cast_to_numpy, *args, **kwargs, ordered=ordered)

    return wrapped


def clone_state(state: eqx.nn.State) -> eqx.nn.State:
    """Clone the state."""
    leaves, treedef = jax.tree.flatten(state)
    state_clone = jax.tree.unflatten(treedef, leaves)
    return state_clone


TFeatures = TypeVar("TFeatures", bound=PyTree[Array])

TObservation = TypeVar("TObservation", bound=PyTree[Array])
TAction = TypeVar("TAction", bound=PyTree[Array])

TObservationSpace = TypeVar("TObservationSpace", bound=spaces.Space)
TActionSpace = TypeVar("TActionSpace", bound=spaces.Space)


def step_env(
    env: Env,
    env_params: gym.EnvParams,
    env_state: gym.EnvState,
    policy: AbstractActorCriticPolicy[
        TFeatures, TObservation, TAction, TObservationSpace, TActionSpace
    ],
    last_policy_state: eqx.nn.State,
    last_observation: TObservation,
    last_done: Bool[ArrayLike, ""],
    *,
    key: Key,
    global_step: Int[ArrayLike, ""],
    tb_writer: SummaryWriter | None = None,
) -> tuple[
    gym.EnvState,
    eqx.nn.State,
    TObservation,
    Bool[Array, ""],
    RolloutBuffer[TObservation, TAction],
]:

    action_key, env_step_key, reset_key = jr.split(key, 3)

    action, value, log_prob, policy_state = policy(
        last_observation, clone_state(last_policy_state), key=action_key
    )

    observation, env_state, reward, done, info = env.step(
        env_step_key, env_state, action, env_params
    )

    if tb_writer is not None:
        tb_log = debug_wrapper(tb_writer.add_scalar)
        tb_log("episode/action", jnp.linalg.norm(action), global_step)
        state = policy.get_hidden_state(policy_state)
        tb_log("episode/state", jnp.linalg.norm(state), global_step)

        def write_info(info: dict[str, Float[Array, ""]]):
            tb_log("episode/reward", info["returned_episode_returns"], global_step)
            tb_log("episode/length", info["returned_episode_lengths"], global_step)

        lax.cond(
            info["returned_episode"],
            write_info,
            lambda _: None,
            info,
        )

    def reset() -> tuple[gym.EnvState, eqx.nn.State, PyTree[Array]]:
        """Reset the environment and policy state."""
        observation, env_state = env.reset(reset_key, env_params)
        new_policy_state = policy.reset(
            env_state, env_params, clone_state(policy_state)
        )

        env_state = fix_env_state(env_state)

        return env_state, new_policy_state, observation

    def identity():
        return env_state, policy_state, observation

    env_state, policy_state, observation = lax.cond(
        done,
        reset,
        identity,
    )

    return (
        env_state,
        policy_state,
        observation,
        done,
        RolloutBuffer(
            observations=last_observation,
            actions=action,
            rewards=reward,
            advantages=jnp.zeros_like(reward),
            returns=jnp.zeros_like(reward),
            terminations=jnp.asarray(last_done),
            truncations=jnp.asarray(False),
            log_probs=log_prob,
            values=value,
            states=last_policy_state,
        ),
    )


def ppo_loss(
    policy: AbstractActorCriticPolicy[
        TFeatures, TObservation, TAction, TObservationSpace, TActionSpace
    ],
    rollout_buffer: RolloutBuffer[TObservation, TAction],
    normalize_advantage: bool,
    clip_value_loss: bool,
    clip_coefficient: float,
    value_coefficient: float,
    state_coefficient: float,
    entropy_coefficient: float,
) -> tuple[
    Float[Array, ""],
    tuple[
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ],
]:
    new_values, new_log_probs, entropy, new_states = jax.vmap(policy.evaluate_action)(
        rollout_buffer.observations, rollout_buffer.actions, rollout_buffer.states
    )

    log_ratio = new_log_probs - rollout_buffer.log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = jnp.mean(ratio - log_ratio) - 1

    advantages = rollout_buffer.advantages
    if normalize_advantage:
        advantages = (advantages - jnp.mean(advantages)) / (
            jnp.std(advantages) + jnp.finfo(advantages.dtype).eps
        )

    policy_loss = -jnp.mean(
        jnp.minimum(
            advantages * ratio,
            advantages * jnp.clip(ratio, 1 - clip_coefficient, 1 + clip_coefficient),
        )
    )

    if clip_value_loss:
        cliped_value = rollout_buffer.values + jnp.clip(
            new_values - rollout_buffer.values, -clip_coefficient, clip_coefficient
        )
        value_loss = (
            jnp.mean(
                jnp.maximum(
                    jnp.square(new_values - rollout_buffer.returns),
                    jnp.square(cliped_value - rollout_buffer.returns),
                )
            )
            / 2
        )
    else:
        value_loss = jnp.mean(jnp.square(new_values - rollout_buffer.returns)) / 2

    entropy_loss = jnp.mean(entropy)

    state_vals = jax.vmap(policy.get_hidden_state)(new_states)
    state_loss = jnp.mean(jnp.linalg.norm(state_vals, axis=-1))

    loss = (
        policy_loss
        + value_coefficient * value_loss
        + state_coefficient * state_loss
        - entropy_coefficient * entropy_loss
    )

    return loss, (loss, policy_loss, value_loss, state_loss, entropy_loss, approx_kl)


ppo_loss_grad = eqx.filter_value_and_grad(ppo_loss, has_aux=True)


class PPO[TFeatures, TObservation, TAction, TObservationSpace, TActionSpace](
    eqx.Module, strict=True
):
    state_index: eqx.nn.StateIndex[
        tuple[
            gym.EnvState,  # Environment state
            PyTree[Array],  # Policy dyanamic values
            eqx.nn.State,  # Policy state
            TObservation,  # Last observation
            Bool[ArrayLike, ""],  # Last done
            Int[ArrayLike, ""],  # Global step
            optax.OptState,  # Optimizer state
        ]
    ]

    policy_static: PyTree = eqx.field(static=True)

    env: Env
    env_params: gym.EnvParams
    observation_space: spaces.Space
    action_space: spaces.Space

    optimizer: optax.GradientTransformation
    learning_rate: float
    anneal_learning_rate: bool

    num_epochs: int
    num_minibatches: int

    total_timesteps: int
    num_steps: int
    gamma: float
    gae_lambda: float
    normalize_advantages: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    value_coefficient: float
    state_coefficient: float
    max_gradient_norm: float

    tb_log_dir: str

    def __init__(
        self,
        policy_class: type[
            AbstractActorCriticPolicy[
                TFeatures, TObservation, TAction, TObservationSpace, TActionSpace
            ]
        ],
        policy_args: tuple,
        policy_kwargs: dict[str, Any],
        env: Env,
        env_params: gym.EnvParams,
        learning_rate: float,
        anneal_learning_rate: bool,
        total_timesteps: int,
        num_steps: int,
        num_epochs: int,
        num_minibatches: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.9,
        normalize_advantages: bool = True,
        clip_coefficient: float = 0.2,
        clip_value_loss: bool = False,
        entropy_coefficient: float = 0.01,
        value_coefficient: float = 0.5,
        state_coefficient: float = 0.5,
        max_gradient_norm: float = 0.5,
        tb_log_dir: str = "runs",
        *,
        key: Key,
    ):
        policy_key, env_reset_key = jr.split(key, 2)

        policy, policy_state = eqx.nn.make_with_state(policy_class)(
            env,
            env_params,
            *policy_args,
            key=policy_key,  # pyright: ignore
            **policy_kwargs,
        )
        policy_dynamic, self.policy_static = eqx.partition(policy, eqx.is_inexact_array)

        self.env = wrappers.LogWrapper(env)  # pyright: ignore
        self.env_params = env_params
        last_observation, env_state = self.env.reset(env_reset_key, env_params)
        last_done = jnp.array(False)
        env_state = fix_env_state(env_state)
        self.observation_space = env.observation_space(env_params)
        self.action_space = env.action_space(env_params)

        if anneal_learning_rate:
            schedule = optax.cosine_decay_schedule(
                learning_rate,
                total_timesteps * num_epochs * num_minibatches // num_steps,
            )
        else:
            schedule = optax.constant_schedule(learning_rate)
        adam = optax.inject_hyperparams(optax.adam)(learning_rate=schedule, eps=1e-5)
        self.optimizer = optax.named_chain(
            ("clipping", optax.clip_by_global_norm(max_gradient_norm)),
            ("adam", adam),
        )
        opt_state = self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array))
        self.learning_rate = learning_rate
        self.anneal_learning_rate = anneal_learning_rate

        self.total_timesteps = total_timesteps
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.clip_coefficient = clip_coefficient
        self.clip_value_loss = clip_value_loss
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.state_coefficient = state_coefficient
        self.max_gradient_norm = max_gradient_norm

        self.tb_log_dir = tb_log_dir

        self.state_index = eqx.nn.StateIndex(
            (
                env_state,
                policy_dynamic,
                policy_state,
                last_observation,
                last_done,
                jnp.array(0),
                opt_state,
            )
        )

    def learn(
        self,
        state: eqx.nn.State,
        tb_log_name: str | None = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        *,
        key: Key,
    ) -> eqx.nn.State:
        if tb_log_name is None:
            tb_writer = None
        else:
            tb_writer = SummaryWriter(f"{self.tb_log_dir}/{tb_log_name}")

        if progress_bar:
            pbar = tqdm(total=self.total_timesteps, desc=f"Training: {tb_log_name}")
            pbar_update = debug_wrapper(pbar.update)

        # Reset the global step to 0 if reset_num_timesteps is True
        state_vals = state.get(self.state_index)
        (
            env_state,
            policy_dynamic,
            policy_state,
            last_observation,
            last_done,
            global_step,
            opt_state,
        ) = state_vals

        if reset_num_timesteps:
            global_step = 0

        state = state.set(
            self.state_index,
            (
                env_state,
                policy_dynamic,
                policy_state,
                last_observation,
                last_done,
                global_step,
                opt_state,
            ),
        )

        @eqx.filter_jit
        def _learn(state):

            def step(val: tuple[eqx.nn.State, Key]) -> tuple[eqx.nn.State, Key]:
                state, key = val
                rollout_key, train_key, carry_key = jr.split(key, 3)
                rollout, state = self.collect_rollout(
                    self.num_steps, state, key=rollout_key, tb_writer=tb_writer
                )
                state = self.train(rollout, state, key=train_key, tb_writer=tb_writer)

                if progress_bar:
                    pbar_update(self.num_steps)  # pyright: ignore

                return state, carry_key

            def cond(val: tuple[eqx.nn.State, Key]) -> Bool[Array, ""]:
                state, _ = val
                global_step = state.get(self.state_index)[5]

                return jnp.asarray(global_step) < jnp.asarray(self.total_timesteps)

            state, _ = lax.while_loop(cond, step, (state, key))

            return state

        state = _learn(state)

        if progress_bar:
            pbar.close()  # pyright: ignore

        return state

    @classmethod
    def load(cls, path: str) -> "PPO":
        # TODO:
        raise NotImplementedError

    def save(self, path: str) -> None:
        # TODO:
        raise NotImplementedError

    def collect_rollout(
        self,
        num_steps: int,
        state: eqx.nn.State,
        *,
        key: Key,
        tb_writer: SummaryWriter | None,
    ) -> tuple[RolloutBuffer[TObservation, TAction], eqx.nn.State]:
        """Collect a rollout from the environment and policy.

        Returns the new state and a rollout buffer.
        """

        (
            env_state,
            policy_dynamic,
            policy_state,
            last_observation,
            last_done,
            global_step,
            opt_state,
        ) = state.get(self.state_index)

        policy: AbstractActorCriticPolicy[
            TFeatures, TObservation, TAction, TObservationSpace, TActionSpace
        ] = eqx.combine(policy_dynamic, self.policy_static)

        def scan_step(
            carry: tuple[
                gym.EnvState,
                eqx.nn.State,
                TObservation,
                Bool[ArrayLike, ""],
                Int[ArrayLike, ""],
                Key,
            ],
            key: Key,
        ) -> tuple[
            tuple[
                gym.EnvState,
                eqx.nn.State,
                TObservation,
                Bool[ArrayLike, ""],
                Int[ArrayLike, ""],
                Key,
            ],
            RolloutBuffer[TObservation, TAction],
        ]:
            (
                env_state,
                policy_state,
                last_observation,
                last_done,
                global_step,
                key,
            ) = carry
            step_key, carry_key = jr.split(key, 2)
            (
                env_state,
                policy_state,
                last_observation,
                last_done,
                rollout_buffer,
            ) = step_env(
                self.env,
                self.env_params,
                env_state,
                policy,  # pyright: ignore
                policy_state,
                last_observation,
                last_done,
                tb_writer=tb_writer,
                global_step=global_step,
                key=step_key,
            )

            return (
                env_state,
                policy_state,
                last_observation,
                last_done,
                global_step + 1,
                carry_key,
            ), rollout_buffer

        (
            env_state,
            policy_state,
            last_observation,
            last_done,
            global_step,
            _,
        ), rollout_buffer = lax.scan(
            scan_step,
            (env_state, policy_state, last_observation, last_done, global_step, key),
            length=num_steps,
        )

        state = state.set(
            self.state_index,
            (
                env_state,
                policy_dynamic,
                policy_state,
                last_observation,
                last_done,
                global_step,
                opt_state,
            ),
        )

        next_value, _ = policy.predict_value(last_observation, policy_state)

        rollout_buffer = compute_returns_and_advantages(
            rollout_buffer, next_value, self.gamma, self.gae_lambda
        )

        return rollout_buffer, state

    def train(
        self,
        rollout_buffer: RolloutBuffer[TObservation, TAction],
        state: eqx.nn.State,
        *,
        tb_writer: SummaryWriter | None,
        key: Key,
    ) -> eqx.nn.State:
        """Train the policy using a collected rollout buffer."""

        (
            env_state,
            policy_dynamic,
            policy_state,
            last_observation,
            last_done,
            global_step,
            opt_state,
        ) = state.get(self.state_index)

        def scan_epoch(
            carry: tuple[
                PyTree,
                optax.OptState,
                Key,
            ],
            _,
        ) -> tuple[
            tuple[
                PyTree,
                optax.OptState,
                Key,
            ],
            None,
        ]:
            def scan_batch(
                carry: tuple[
                    PyTree,
                    optax.OptState,
                ],
                rollout_buffer: RolloutBuffer[TObservation, TAction],
            ) -> tuple[
                tuple[
                    PyTree,
                    optax.OptState,
                ],
                Any,
            ]:
                policy_dynamic, opt_state = carry

                policy: AbstractActorCriticPolicy[
                    TFeatures, TObservation, TAction, TObservationSpace, TActionSpace
                ] = eqx.combine(policy_dynamic, self.policy_static)

                (_, stats), grads = ppo_loss_grad(
                    policy,  # pyright: ignore
                    rollout_buffer,
                    self.normalize_advantages,
                    self.clip_value_loss,
                    self.clip_coefficient,
                    self.value_coefficient,
                    self.state_coefficient,
                    self.entropy_coefficient,
                )

                flat_grads = jnp.concatenate(
                    jax.tree.flatten(jax.tree.map(jnp.ravel, grads))[0]
                )

                def apply_gradients():
                    updates, _opt_state = self.optimizer.update(grads, opt_state)
                    policy_dynamic, _ = eqx.partition(
                        eqx.apply_updates(policy, updates), eqx.is_array
                    )

                    return policy_dynamic, _opt_state

                def identity():
                    return policy_dynamic, opt_state

                policy_dynamic, opt_state = lax.cond(
                    jnp.any(jnp.isnan(flat_grads)), identity, apply_gradients
                )

                return (policy_dynamic, opt_state), stats

            policy_dynamic, opt_state, key = carry
            batch_key, carry_key = jr.split(key, 2)
            (policy_dynamic, opt_state), stats = lax.scan(
                scan_batch,
                (policy_dynamic, opt_state),
                batches(
                    rollout_buffer,
                    self.num_steps // self.num_minibatches,
                    key=batch_key,
                ),
            )

            stats = jax.tree.map(jnp.mean, stats)

            return (policy_dynamic, opt_state, carry_key), stats

        (policy_dynamic, opt_state, _), stats = lax.scan(
            scan_epoch,
            (policy_dynamic, opt_state, key),
            length=self.num_epochs,
        )

        stats = jax.tree.map(jnp.mean, stats)

        state = state.set(
            self.state_index,
            (
                env_state,
                policy_dynamic,
                policy_state,
                last_observation,
                last_done,
                global_step,
                opt_state,
            ),
        )

        learning_rate = opt_state["adam"].hyperparams[  # pyright: ignore
            "learning_rate"
        ]

        variance = jnp.var(rollout_buffer.rewards)
        explained_variance = 1 - jnp.var(
            rollout_buffer.returns - rollout_buffer.values
        ) / (variance + jnp.finfo(variance.dtype).eps)

        if tb_writer is not None:
            tb_log = debug_wrapper(tb_writer.add_scalar, ordered=True)
            tb_log("loss/total", stats[0], global_step)
            tb_log("loss/policy", stats[1], global_step)
            tb_log("loss/value", stats[2], global_step)
            tb_log("loss/state", stats[3], global_step)
            tb_log("loss/entropy", stats[4], global_step)
            tb_log("loss/approx_kl", stats[5], global_step)
            tb_log("loss/learning_rate", learning_rate, global_step)
            tb_log("stats/variance", variance, global_step)
            tb_log("stats/explained_variance", explained_variance, global_step)

        return state

    def policy(self, state: eqx.nn.State):
        """Return the policy."""
        return eqx.combine(state.get(self.state_index)[1], self.policy_static)
