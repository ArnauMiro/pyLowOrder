# This file serves as an example of how to use Optuna for hyperparameter optimization
# in Reinforcement Learning with Stable-Baselines3 and pyLOM.
# It optimizes the hyperparameters of a PPO agent on the 'neuralfoil' environment.

from copy import deepcopy

import pyLOM
try: 
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
except:
    import sys
    pyLOM.pprint(0, "To run this example you need to have optuna intalled. Please, intall it with pip install optuna")
    sys.exit(1)

import gymnasium as gym
import optuna
import torch
import torch.nn as nn
import pyLOM.RL 
from stable_baselines3.common.callbacks import EvalCallback

# Set pytorch num threads to 1 for faster training. This is because torch parallelize the network (in both, neuralfoil and PPO networks) inference with all available cpus,
# which in some cases and with small networks can lead to slower training.
torch.set_num_threads(1)

N_TRIALS = 1000  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 10  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 10  # Number of evaluations during the training
N_TIMESTEPS = 25000  # Training budget
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 80
TIMEOUT = int(60 * 15)  # 15 minutes

TRAIN_ENV = pyLOM.RL.create_env('neuralfoil')

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": TRAIN_ENV,
}

def sample_ppo_params(trial):

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.3, 0.5, 0.75, 0.9, 0.95, 0.99])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.001, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.05, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [10, 20, 30])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 1, 2])
    vf_coef = trial.suggest_float("vf_coef", 0.2, 0.8)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if batch_size > n_steps:
        batch_size = n_steps

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

class TrialEvalCallback(EvalCallback):

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
    
def objective(trial) -> float:
    kwargs = deepcopy(DEFAULT_HYPERPARAMS)

    # 1. Sample hyperparameters and update the default keyword arguments: `kwargs.update(other_params)`
    sampled_params = sample_ppo_params(trial)
    kwargs.update(sampled_params)
    # 2. Create the evaluation envs
    eval_envs = pyLOM.RL.create_env('neuralfoil', num_envs=1)
    # 3. Create the `TrialEvalCallback`
    eval_callback = TrialEvalCallback(
        eval_env=eval_envs,
        trial=trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
    )
    # Create the RL model
    model = pyLOM.RL.SB3_PPO(**kwargs)

    nan_encountered = False
    try:
        # Train the model
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        pyLOM.pprint(0, e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
# Do not prune before 1/3 of the max budget is used
pruner = MedianPruner(
    n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
)
# Create the study and start the hyperparameter optimization
study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

try:
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
except KeyboardInterrupt:
    pass

pyLOM.pprint(0, "Number of finished trials: ", len(study.trials))

pyLOM.pprint(0, "Best trial:")
trial = study.best_trial

pyLOM.pprint(0, f"  Value: {trial.value}")

pyLOM.pprint(0, "  Params: ")
for key, value in trial.params.items():
    pyLOM.pprint(0, f"    {key}: {value}")

pyLOM.pprint(0, "  User attrs:")
for key, value in trial.user_attrs.items():
    pyLOM.pprint(0, f"    {key}: {value}")

# Write report
study.trials_dataframe().to_csv("study_results_ppo_optuna.csv")

fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)
fig3 = plot_parallel_coordinate(study)
# save the Figures
fig1.write_image("optuna_optimization_history.png")
fig2.write_image("optuna_param_importances.png")
fig3.write_image("optuna_parallel_coordinate.png")
# save as html too
fig1.write_html("optuna_optimization_history.html")
fig2.write_html("optuna_param_importances.html")
fig3.write_html("optuna_parallel_coordinate.html")
fig1.show()
fig2.show()
fig3.show()