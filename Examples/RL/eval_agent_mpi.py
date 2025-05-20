# This script is used to evaluate a trained agent on th UIUC airfoil dataset.
# In the exmple the agent is loaded from a file, to train an agent 
# you can use the train_agent_airfoil.py script 
# The evaluation is done with xfoil and is accelerated with MPI.
# To run this, first, you need to make sure that xfoil is intalled 
# (https://github.com/RobotLocomotion/xfoil/) and on the $PATH
# Then, run on your terminal: mpirun -np <number of mpi processes> eval_agent_mpi.py 

from pyLOM.RL import create_env, AirfoilOperatingConditions, NON_CONVERGED_REWARD
from pyLOM.RL.evaluations import evaluate_airfoil_agent_whole_uiuc_mpi
from pyLOM.utils import pprint

from stable_baselines3 import PPO
import pandas as pd
import numpy as np
from mpi4py import MPI

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env = create_env(
        "xfoil",
        operating_conditions=AirfoilOperatingConditions(alpha=4.0, mach=0.2, Reynolds=1e6),
        episode_max_length=64,
        thickness_penalization_factor=0.0
    )
    # Load the agent that was already trained
    agent = PPO.load("airfoil_agent", env=env, device="cpu")
    # Evaluate the agent on the UIUC airfoil dataset
    evaluate_airfoil_agent_whole_uiuc_mpi(agent, env, save_results_path="results.csv")

    # show metrics extraced from the results
    if rank == 0:
        df = pd.read_csv("results.csv")
        # filter the states that didn't converge.
        df = df[(df['best_CLCD'] > NON_CONVERGED_REWARD) & (df['initial_CLCD'] > NON_CONVERGED_REWARD)]
        df = df[df['initial_CLCD'] != df['best_CLCD']]
        best_rewards = df["best_CLCD"].values
        initial_rewards = df["initial_CLCD"].values
        q75, q25 = np.percentile(best_rewards, [75 ,25])
        pprint(0, f"Number of airfoils evaluated: {len(df)}")
        pprint(0, f"Best CL/CD [median(IQR)]: {round(np.median(best_rewards))} ({round(q75 - q25)})")
        pprint(0, f"Best CL/CD increment [mean +/- std]: {(best_rewards - initial_rewards).mean():.1f}+/-{(best_rewards - initial_rewards).std():.1f}")
        