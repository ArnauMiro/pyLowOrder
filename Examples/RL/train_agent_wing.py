# This script shows an example of how to train an agent using 
# the PPO algorithm from Stable Baselines3. The solver used is AeroBuildup (ptLOM.RL.AerosandboxWingSolver)
# and the space state bounds used are the default ones.
# This example is very similar to the airfoil training example, 
# only a few lines are different. That's one of the strengths of pyLOM.
import pyLOM.RL
from stable_baselines3.common.evaluation import evaluate_policy
import torch

# Set pytorch num threads to 1 for faster training. This is because torch parallelize the network inference with all available cpus,
# which in some cases and with small networks can lead to slower training.
torch.set_num_threads(1)

N_ENVS = 8
N_TIMESTEPS = 15_000

ppo_params = {
    'learning_rate': 2.5e-4,
    'n_steps': 2048 // N_ENVS, 
    'batch_size': 32,
    'n_epochs': 20,
    'gamma': 0.3,
    'gae_lambda': 0.95,
    'clip_range': 0.4,
    'ent_coef': 0.005,
    'verbose': 1,
    'policy_kwargs': {'net_arch': dict(pi=[256, 256], vf=[256, 256])},
}

if __name__ == "__main__":
    operating_conditions = pyLOM.RL.WingOperatingConditions(
        alpha=2.0,
        altitude=500,
        velocity=200,
    )

    # Create the environment
    env = pyLOM.RL.create_env(
        solver_name="aerosandbox",
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
    model.save("wing_agent")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    pyLOM.pprint(0, f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")