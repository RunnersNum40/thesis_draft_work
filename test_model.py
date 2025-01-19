from stable_baselines3 import PPO

from cpg_env_ppo import BEST_MODEL_PATH, env_fn
from test import visualize_run

EPISODE_STEPS = 600

env = env_fn()
model = PPO.load(BEST_MODEL_PATH)


visualize_run(env, model, EPISODE_STEPS, 0)
