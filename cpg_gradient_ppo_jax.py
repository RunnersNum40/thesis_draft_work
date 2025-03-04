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
from jax import Array, jr
from torch.utils.tensorboard import SummaryWriter  # pyright: ignore
from tqdm import TqdmExperimentalWarning, trange

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
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Ant-v5"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    convergence_factor: float = 1000.0
    """convergence factor of the CPG model"""
    num_oscillators: int = 4
    """number of oscillators in the CPG model"""
    timestep: float = 1e-2
    """time increment for CPG steps"""
    solver: str = "rk4"
    """ode solver to use with the ODE"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    def __hash__(self):
        return hash(frozenset(vars(self).items()))


class ContinuousAgent(nnx.Module):
    def __init__(self, env: gym.Env, *, rngs: nnx.Rngs) -> None:
        critic_layers = [np.array(env.observation_space.shape).prod(), 64, 64, 1]
        actor_layers = [
            int(np.array(env.observation_space.shape).prod()),
            64,
            64,
            int(np.array(env.action_space.shape).prod()),
        ]

        self.critic = MLP(critic_layers, nnx.tanh, 1.0, rngs=rngs)
        self.actor = MLP(actor_layers, nnx.relu, 0.01, rngs=rngs)
        self.actor_logstd = nnx.Variable(
            jnp.zeros(jnp.asarray(env.action_space.shape).prod())
        )

    @nnx.jit
    def get_value(self, x: Array) -> Array:
        return jnp.squeeze(self.critic(x))

    @nnx.jit
    def get_action_and_value(
        self,
        x: Array,
        key: Array | None,
        action: Array | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        action_mean = self.actor(x)
        action_std = jnp.exp(self.actor_logstd.value)

        if action is None:
            if key is not None:
                action = action_mean + action_std * jr.normal(key, action_mean.shape)
            else:
                action = action_mean

        log_prob = jax.scipy.stats.norm.logpdf(
            action, loc=action_mean, scale=action_std
        ).sum()

        entropy = 0.5 * (jnp.log(2 * jnp.pi * action_std**2) + 1)

        value = jnp.squeeze(self.critic(x))

        return action, log_prob, entropy, value


@nnx.jit(static_argnames=["args"])
def compute_gae(
    agent: ContinuousAgent,
    next_observation: Array,
    next_done: Array,
    rewards: Array,
    values: Array,
    dones: Array,
    args: Args,
) -> tuple[Array, Array]:
    next_value = agent.get_value(next_observation)

    next_values = jnp.concatenate([values[1:], next_value[None]], axis=0)
    next_non_terminal = jnp.concatenate(
        [1.0 - dones[1:], jnp.array([1.0 - next_done], dtype=rewards.dtype)], axis=0
    )

    def scan_fn(carry, x):
        reward, value, next_val, non_terminal = x
        delta = reward + args.gamma * next_val * non_terminal - value
        new_carry = delta + args.gamma * args.gae_lambda * non_terminal * carry
        return new_carry, new_carry

    inputs = (
        jnp.flip(rewards, axis=0),
        jnp.flip(values, axis=0),
        jnp.flip(next_values, axis=0),
        jnp.flip(next_non_terminal, axis=0),
    )
    inputs_stacked = jnp.stack(inputs, axis=1)
    init_carry = jnp.array(0.0, dtype=rewards.dtype)
    _, advantages_rev = jax.lax.scan(scan_fn, init_carry, inputs_stacked)

    advantages = jnp.flip(advantages_rev, axis=0)
    returns = advantages + values

    return advantages, returns


@nnx.jit(static_argnames=["args"])
def ppo_loss(
    agent: ContinuousAgent,
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
    agent: ContinuousAgent,
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
    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    logger.info(f"Run name: {run_name}")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    np.jr.seed(args.seed)
    key = jr.key(args.seed)
    key, flax_rngs = jr.split(key)
    flax_rngs = nnx.Rngs(key)

    if args.capture_video:
        env = gym.make(args.env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert isinstance(env.observation_space, gym.spaces.Box)

    agent = ContinuousAgent(env, rngs=flax_rngs)
    adam = optax.adam(
        learning_rate=(
            optax.cosine_decay_schedule(args.learning_rate, args.total_timesteps)
            if args.anneal_lr
            else args.learning_rate
        )
    )
    optimizer = nnx.Optimizer(
        agent, optax.chain(optax.clip_by_global_norm(args.max_grad_norm), adam)
    )
    observations = jnp.zeros((args.num_steps,) + env.observation_space.shape)
    actions = jnp.zeros((args.num_steps,) + env.action_space.shape)
    log_probs = jnp.zeros((args.num_steps,))
    rewards = jnp.zeros((args.num_steps,))
    dones = jnp.zeros((args.num_steps,))
    values = jnp.zeros((args.num_steps,))

    global_step = 0
    next_observation, _ = env.reset(seed=args.seed)
    next_observation = jnp.asarray(next_observation)
    next_done = jnp.array(0)

    loss = jnp.nan
    policy_loss = jnp.nan
    value_loss = jnp.nan
    entropy_loss = jnp.nan
    approx_kl = jnp.nan

    for iteration in trange(1, args.num_iterations + 1, desc="Iterations"):
        for step in trange(args.num_steps, leave=False, desc="Steps"):
            global_step += 1
            observations = observations.at[step].set(next_observation)
            dones = dones.at[step].set(next_done)

            key, subkey = jr.split(key)
            action, log_prob, _, value = agent.get_action_and_value(
                next_observation, subkey
            )
            values = values.at[step].set(value)
            actions = actions.at[step].set(action)
            log_probs = log_probs.at[step].set(log_prob)

            next_observation, reward, termination, truncation, info = env.step(action)
            rewards = rewards.at[step].set(reward)
            next_observation = jnp.asarray(next_observation)
            next_done = jnp.asarray(int(termination or truncation))

            if "episode" in info:
                writer.add_scalar("episode/reward", info["episode"]["r"], global_step)
                writer.add_scalar("episode/length", info["episode"]["l"], global_step)
                writer.add_scalar("episode/time", info["episode"]["t"], global_step)

            if termination or truncation:
                next_observation, _ = env.reset(seed=args.seed)
                next_observation = jnp.array(next_observation)

        advantages, returns = compute_gae(
            agent, next_observation, next_done, rewards, values, dones, args
        )

        for epoch in trange(args.update_epochs, leave=False, desc="Epochs"):
            key, subkey = jr.split(key)
            batch_indices = jr.permutation(subkey, args.batch_size, independent=True)
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
