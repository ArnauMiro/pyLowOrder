# This script shows an example of how to train an agent using 
# the PPO algorithm from Stable Baselines3. The solver used is neuralfoil
# and the space state bounds used are the default ones.

import pyLOM.RL
import torch

# Set pytorch num threads to 1 for faster training. This is because torch parallelize the network inference (in both, neuralfoil and PPO networks) with all available cpus,
# which in some cases and with small networks can lead to slower training.
torch.set_num_threads(1)

N_ENVS = 4
N_TIMESTEPS = 50_000

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

if __name__ == "__main__":
    operating_conditions = pyLOM.RL.AirfoilOperatingConditions(
        alpha=4.0,
        mach=0.2,
        Reynolds=1e6,
    )

    # Create the environment
    env = pyLOM.RL.create_env(
        solver_name="neuralfoil",
        operating_conditions=operating_conditions,
        num_envs=N_ENVS,
        episode_max_length=64,
        thickness_penalization_factor=0.0
    )

    # Instantiate and train the model. Here, PPO from Stable Baselines3 is used,
    # but you can use any other RL algorithm from Stable Baselines3 or any other library compatible with Gymmasyum.
    model = pyLOM.RL.SB3_PPO("MlpPolicy", env, **ppo_params)
    model.learn(total_timesteps=N_TIMESTEPS)

    # Save the model
    model.save("airfoil_agent")