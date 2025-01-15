import gymnasium as gym
from stable_baselines3 import PPO

import cpg_env
from visualization import animate_trajectory, plot_polar_trajectory, plot_trajectory

EPISODE_STEPS = 100
NUM_EPISODES = 100
TIMESTEPS = EPISODE_STEPS * NUM_EPISODES

env = gym.make("SquareCPGEnv-v0", max_episode_steps=EPISODE_STEPS, n=1)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="runs/",
)

model.learn(total_timesteps=TIMESTEPS)

model.save("ppo_cpg_square")


def visualize_run(env, model, steps=1000):
    obs, _ = env.reset()
    states = []
    params = []

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        states.append(env.unwrapped.state)
        params.append(cpg_env.action_to_params(action))

        if done:
            break

    states_and_params = list(zip(states, params))
    plot_trajectory(states_and_params, env.unwrapped.dt)
    plot_polar_trajectory(states_and_params)
    animate_trajectory(states_and_params)


visualize_run(env, model, EPISODE_STEPS)
