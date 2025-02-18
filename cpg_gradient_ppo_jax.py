import logging
import os
import time
import warnings
from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import nnx
from jax import Array, random
from torch.utils.tensorboard import SummaryWriter
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from cpg_gradient_jax import MLP, CPGNetwork

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
)
os.environ["TF_CUDNN DETERMINISTIC"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1  # TODO: Implement parallel environments
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    def __hash__(self):
        return hash(frozenset(vars(self).items()))


class DiscreteAgent(nnx.Module):
    def __init__(self, env: gym.Env, *, rngs: nnx.Rngs) -> None:
        critic_layers = [np.array(env.observation_space.shape).prod(), 64, 64, 1]
        actor_layers = [
            np.array(env.observation_space.shape).prod(),
            64,
            64,
            env.action_space.n,  # pyright: ignore
        ]

        self.critic = MLP(critic_layers, nnx.relu, 1.0, rngs=rngs)
        self.actor = MLP(actor_layers, nnx.relu, 0.01, rngs=rngs)

    @nnx.jit
    def get_value(self, x: Array) -> Array:
        return jnp.squeeze(self.critic(x))

    @nnx.jit
    def get_action(self, x: Array, key: Array | None) -> Array:
        logits = self.actor(x)
        if key is None:
            return jnp.argmax(logits, axis=-1)
        return random.categorical(key, logits=logits)

    @nnx.jit
    def get_action_and_value(
        self,
        x: Array,
        key: Array | None,
        action: Array | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        logits = self.actor(x)
        if action is None:
            if key is None:
                action = jnp.argmax(logits, axis=-1)
            else:
                action = random.categorical(key, logits=logits)

        log_prob = nnx.log_softmax(logits)[action]
        entropy = -jnp.sum(jnp.exp(log_prob) * log_prob)
        return action, log_prob, entropy, self.get_value(x)


@nnx.jit(static_argnames=["args"])
def compute_gae(
    agent: DiscreteAgent,
    next_observation: Array,
    next_done: Array,
    rewards: Array,
    values: Array,
    dones: Array,
    args: Args,
) -> tuple[Array, Array]:
    next_value = agent.get_value(next_observation)
    advantages = jnp.zeros_like(rewards)
    last_gae_lambda = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            next_non_terminal = 1.0 - next_done
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
        last_gae_lambda = (
            delta + args.gamma * args.gae_lambda * last_gae_lambda * next_non_terminal
        )
        advantages = advantages.at[t].set(last_gae_lambda)

    returns = advantages + values

    return advantages, returns


@nnx.jit(static_argnames=["args"])
def ppo_loss(
    agent: DiscreteAgent,
    observations: Array,
    actions: Array,
    advantages: Array,
    returns: Array,
    values: Array,
    log_probs: Array,
    args: Args,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    _, new_log_prob, entropy, new_value = jax.vmap(agent.get_action_and_value)(
        observations, None, actions
    )
    log_ratio = new_log_prob - log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = jnp.mean(((ratio - 1) - log_ratio))

    if args.norm_adv:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    policy_loss = jnp.mean(
        jnp.maximum(
            -advantages * ratio,
            -advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
        )
    )

    if args.clip_vloss:
        value_loss_unclipped = (returns - new_value) ** 2
        value_clipped = values + jnp.clip(
            new_value - values, -args.clip_coef, args.clip_coef
        )
        value_loss_clipped = (returns - value_clipped) ** 2
        value_loss = (
            jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped)) / 2.0
        )
    else:
        value_loss = jnp.mean((returns - new_value) ** 2) / 2.0

    entropy_loss = jnp.mean(entropy)
    loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
    return loss, (
        policy_loss,
        entropy_loss,
        value_loss,
        jax.lax.stop_gradient(approx_kl),
    )


ppo_grad = nnx.value_and_grad(ppo_loss, has_aux=True)


@nnx.jit(static_argnames=["args"])
def train_batch(
    agent: DiscreteAgent,
    optimizer: nnx.Optimizer,
    observations: Array,
    actions: Array,
    advantages: Array,
    returns: Array,
    values: Array,
    log_probs: Array,
    args: Args,
):
    observations = jax.lax.stop_gradient(observations)
    actions = jax.lax.stop_gradient(actions)
    advantages = jax.lax.stop_gradient(advantages)
    returns = jax.lax.stop_gradient(returns)
    log_probs = jax.lax.stop_gradient(log_probs)

    (loss, (policy_loss, value_loss, entropy_loss, approx_kl)), grads = ppo_grad(
        agent, observations, actions, advantages, returns, values, log_probs, args
    )
    optimizer.update(grads)
    return loss, policy_loss, value_loss, entropy_loss, approx_kl


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    np.random.seed(args.seed)
    key = random.key(args.seed)
    flax_rngs = nnx.Rngs(key)

    if args.capture_video:
        env = gym.make(args.env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    agent = DiscreteAgent(env, rngs=flax_rngs)
    adam = optax.adam(
        learning_rate=(
            optax.cosine_decay_schedule(args.learning_rate, args.total_timesteps)
            if args.anneal_lr
            else args.learning_rate
        )
    )
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), adam)
    optimizer = nnx.Optimizer(agent, optimizer)
    observations = jnp.zeros((args.num_steps,) + env.observation_space.shape)
    actions = jnp.zeros((args.num_steps,), dtype=jnp.int32)
    log_probs = jnp.zeros((args.num_steps,))
    rewards = jnp.zeros((args.num_steps,))
    dones = jnp.zeros((args.num_steps,))
    values = jnp.zeros((args.num_steps,))

    global_step = 0
    start_time = time.time()
    next_observation, _ = env.reset(seed=args.seed)
    next_observation = jnp.array(next_observation)
    next_done = jnp.array(0)

    for iteration in tqdm(range(1, args.num_iterations + 1), desc="Iterations"):
        for step in range(args.num_steps):
            global_step += 1
            observations = observations.at[step].set(next_observation)
            dones = dones.at[step].set(next_done)

            key, subkey = random.split(key)
            action, log_prob, entropy, value = agent.get_action_and_value(
                next_observation, subkey
            )
            values = values.at[step].set(value)
            actions = actions.at[step].set(action)
            log_probs = log_probs.at[step].set(log_prob)

            next_observation, reward, termination, truncation, info = env.step(
                action.item()
            )
            rewards = rewards.at[step].set(reward)
            next_observation = jnp.array(next_observation)
            next_done = jnp.array(int(termination or truncation))

            if info:
                writer.add_scalar("episode/reward", info["episode"]["r"], global_step)
                writer.add_scalar("episode/length", info["episode"]["l"], global_step)
                writer.add_scalar("episode/time", info["episode"]["t"], global_step)

            if termination or truncation:
                next_observation, _ = env.reset(seed=args.seed)
                next_observation = jnp.array(next_observation)

        advantages, returns = compute_gae(
            agent, next_observation, next_done, rewards, values, dones, args
        )

        for epoch in range(args.update_epochs):
            key, subkey = random.split(key)
            batch_indices = random.permutation(
                subkey, args.batch_size, independent=True
            )
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = batch_indices[start:end]

                loss, policy_loss, value_loss, entropy_loss, approx_kl = train_batch(
                    agent,
                    optimizer,
                    observations[mb_inds],
                    actions[mb_inds],
                    advantages[mb_inds],
                    returns[mb_inds],
                    values[mb_inds],
                    log_probs[mb_inds],
                    args,
                )

            if args.target_kl is not None and float(approx_kl) > args.target_kl:
                break

        variance = jnp.var(returns)
        explained_variance = (
            jnp.nan if variance == 0 else 1 - jnp.var(values - returns) / variance
        )

        writer.add_scalar("loss/total", float(loss), global_step)
        writer.add_scalar("loss/policy", float(policy_loss), global_step)
        writer.add_scalar("loss/value", float(value_loss), global_step)
        writer.add_scalar("loss/entropy", float(entropy_loss), global_step)
        writer.add_scalar("loss/kl", float(approx_kl), global_step)
        writer.add_scalar(
            "loss/explained_variance", float(explained_variance), global_step
        )
