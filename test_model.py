from stable_baselines3 import PPO

from cpg_env_ppo import BEST_MODEL_PATH, make_env
from test import visualize_run
import seaborn as sns

sns.set_theme(style="whitegrid")

BEST_MODEL_PATH = "checkpoints-ellipse/best_model.zip"

EPISODE_STEPS = 1000

env_fn = make_env(
    "SquareCPGEnv-v0",
    n=1,
    state_noise=0.01,
    observation_noise=0.01,
    action_noise=0.01,
    observe_actions=1,
)

env = env_fn()
model = PPO.load(BEST_MODEL_PATH)


visualize_run(env, model, EPISODE_STEPS, 0)
