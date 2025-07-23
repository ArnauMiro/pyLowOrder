from pyLOM.RL import create_env, AirfoilOperatingConditions
from stable_baselines3 import PPO
import torch

# Set pytorch num threads to 1 for faster training
torch.set_num_threads(1)

N_ENVS = 4
N_TIMESTEPS = 10_000

ppo_params = {
    "learning_rate": 2.5e-4,
    "n_steps": 2048 // N_ENVS,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.3,
    "ent_coef": 0, 
    "clip_range": 0.6,
    "verbose": 1,
    "device": "cpu",
    "policy_kwargs": {"net_arch": dict(pi=[256, 256], vf=[256, 256])},
}

operating_conditions = AirfoilOperatingConditions(
    alpha=4.0,
    mach=0.2,
    Reynolds=1e6,
)

# Create the environment
env = create_env(
    solver_name="neuralfoil",
    operating_conditions=operating_conditions,
    num_envs=N_ENVS,
    episode_max_length=64,
    thickness_penalization_factor=0.0
)

# Define the model
model = PPO("MlpPolicy", env, **ppo_params)
model.learn(total_timesteps=N_TIMESTEPS)

# Save the model
model.save("airfoil_agent")