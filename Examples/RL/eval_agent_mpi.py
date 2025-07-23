# This script is used to evaluate a trained agent on th UIUC airfoil dataset.
# In the exmple the agent is loaded from a file, to train an agent 
# you can use the train_agent_airfoil.py script 
# The evaluation is done with xfoil and is accelerated with MPI.
# To run this, first, you need to make sure that xfoil is intalled 
# (https://github.com/RobotLocomotion/xfoil/) and on the $PATH
# Then, run on your terminal: mpirun -np <number of mpi processes> eval_agent_mpi.py 

import pyLOM.RL


# Create the environment. The environment must be created with the same parameters as the one used to train the agent, except for the solver name.
env = pyLOM.RL.create_env(
    "xfoil",
    operating_conditions=pyLOM.RL.AirfoilOperatingConditions(
        alpha=4.0, mach=0.2, Reynolds=1e6
    ),
    episode_max_length=64,
    thickness_penalization_factor=0.0,
)
# Load the agent that was already trained
agent = pyLOM.RL.SB3_PPO.load("airfoil_agent", env=env, device="cpu")
# Evaluate the agent on the UIUC airfoil dataset. This will print a summary of the results and save them to a CSV file.
pyLOM.RL.evaluate_airfoil_agent_whole_uiuc_mpi(agent, env, save_results_path="results.csv")