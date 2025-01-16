import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import cpg_env
from visualization import animate_trajectory, plot_polar_trajectory, plot_trajectory

NUM_ENVS = 6
EPISODE_STEPS = 1000
NUM_EPISODES = 500
TIMESTEPS = EPISODE_STEPS * NUM_EPISODES


def make_env(env_id: str, *args, **kwargs):
    def _init():
        env = gym.make(env_id, *args, **kwargs)
        env = Monitor(env)
        return env

    return _init


env_fns = [
    make_env("SquareCPGEnv-v0", max_episode_steps=EPISODE_STEPS, n=1)
    for i in range(NUM_ENVS)
]

vec_env = SubprocVecEnv(env_fns, start_method="fork")


model = PPO(
    policy="MlpPolicy",
    env=vec_env,
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


env = gym.make("SquareCPGEnv-v0", n=1)
visualize_run(env, model, EPISODE_STEPS)
