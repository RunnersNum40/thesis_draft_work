import equinox as eqx
from jax import random as jr
from jax import nn as jnn

import gymnax
from ppo import PPO
from policies import CDEActorMLPCriticPolicy
import wrappers


def main():
    env_id = "Pendulum-v1"
    key = jr.key(0)

    env, env_params = gymnax.make(env_id)
    env = wrappers.ClipActionWrapper(env)
    env = wrappers.RescaleAction(env)
    env = wrappers.AddTimeWrapper(env)

    learning_rate = 3e-4
    num_steps = 1024
    num_epochs = 8
    num_minibatches = 32
    total_timesteps = 1048576

    width_size = 64
    depth = 2

    ppo_agent, state = eqx.nn.make_with_state(PPO)(
        policy_class=CDEActorMLPCriticPolicy,
        policy_args=(),
        policy_kwargs={
            "width_size": width_size,
            "depth": depth,
            "state_size": 4,
            "max_steps": 4,
        },
        env=env,
        env_params=env_params,
        learning_rate=learning_rate,
        anneal_learning_rate=True,
        num_steps=num_steps,
        num_epochs=num_epochs,
        state_coefficient=0.1,
        max_gradient_norm=1.0,
        num_minibatches=num_minibatches,
        key=key,
    )

    state = ppo_agent.learn(
        state,
        total_timesteps=total_timesteps,
        tb_log_name=f"{env_id}_mlp_critic",
        key=key,
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
