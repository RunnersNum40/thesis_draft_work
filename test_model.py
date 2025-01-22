import seaborn as sns
from stable_baselines3 import PPO

from cpg_env_ppo import BEST_MODEL_PATH, make_env
from test import rollout_run, visualize_run

sns.set_theme(style="whitegrid")

BEST_MODEL_PATH = "checkpoints-square/best_model.zip"

EPISODE_STEPS = 500

env_fn = make_env(
    "SquareCPGEnv-v0",
    n=1,
    state_noise=0.0,
    observation_noise=0.0,
    action_noise=0.0,
    time_step_range=(10, 10),
    observe_actions=1,
)

env = env_fn()
model = PPO.load(BEST_MODEL_PATH)


reward, states, params = rollout_run(env, model, EPISODE_STEPS, 0)
print(f"Reward {reward}")
visualize_run(env, states, params, show_params=False)
