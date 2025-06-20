import pyLOM.RL
import torch

# Set pytorch num threads to 1 for faster training. This is because torch parallelize the network (in both, neuralfoil and PPO networks) inference with all available cpus,
# which in some cases and with small networks can lead to slower training.
torch.set_num_threads(1)

# Configuration parameters
NUM_ENVS = 8
NEURALFOIL_ITERATIONS = 25000
XFOIL_ITERATIONS = 15000
THICKNESS_PENALIZATION = 0

if __name__ == "__main__":
    first_env = pyLOM.RL.create_env(
        "neuralfoil", thickness_penalization_factor=THICKNESS_PENALIZATION
    )
    neuralfoil_hyperparameters = {
        "learning_rate": 2.5e-4, 
        "n_steps": 2048, 
        "batch_size": 64, 
        "n_epochs": 10,
        "gamma": 0.25,
        "ent_coef": 0, 
        "clip_range": 0.6,
        "verbose": 1,
        "policy_kwargs": {"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    }
    initial_model = pyLOM.RL.SB3_PPO(
        "MlpPolicy", first_env, **neuralfoil_hyperparameters
    )
    initial_model.learn(total_timesteps=NEURALFOIL_ITERATIONS)
    initial_model.save("pretrained_model")
    initial_policy = initial_model.policy

    second_env = pyLOM.RL.create_env(
        "xfoil", num_envs=NUM_ENVS, thickness_penalization_factor=THICKNESS_PENALIZATION
    )
    xfoil_hyperparameters = {
        "learning_rate": 2.5e-4,
        "n_steps": 512 // NUM_ENVS,
        "batch_size": 64,
        "n_epochs": 20,
        "gamma": 0.3,
        "ent_coef": 0.005, 
        "verbose": 1,
        "policy_kwargs": {"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    }
    fine_tuned_model = pyLOM.RL.SB3_PPO(
        "MlpPolicy", second_env, device='cpu', **xfoil_hyperparameters
    )
    # load the weights of the initial policy
    fine_tuned_model.policy.load_state_dict(initial_policy.state_dict())
    fine_tuned_model.learn(total_timesteps=XFOIL_ITERATIONS)
    fine_tuned_model.save("fine_tuned_model")