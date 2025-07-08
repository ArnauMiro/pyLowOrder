import os
import random

import aerosandbox as asb
import numpy as np
from aerosandbox import _asb_root

from pyLOM.utils import pprint
from pyLOM.RL import NON_CONVERGED_REWARD
from pyLOM.utils.mpi import MPI_RANK, MPI_SIZE, MPI_Status, MPI_ANY_TAG, MPI_ANY_SOURCE, mpi_send, mpi_recv
import pyLOM

airfoil_database_root = _asb_root / "geometry" / "airfoil" / "airfoil_database"

def run_episode(rl_model, env, initial_shape=None, seed=None, keep_unconverged=False):
    """
    Runs a single episode of the RL agent in the given environment.

    Args:
        rl_model: The RL model to evaluate.
        env: The environment in which to run the episode.
        initial_shape: An initial airfoil shape to start the episode with. Default: ``None``, which means a random shape will be generated.
        seed: A seed for reproducibility. Default: ``None``.
        keep_unconverged: If True, keeps the last state and reward even if the episode was truncated. Default: ``False``.

    Returns:
        rewards: A list of cumulative rewards at each step.
        states: A list of airfoil shapes at each step.
    """
    done = False
    truncated = False
    states = []
    rewards = []
    
    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        env.action_space.seed(seed)
    reset_options = {"initial_shape": initial_shape} if initial_shape is not None else {}
    obs, info = env.reset(seed=seed, options=reset_options)
    initial_reward = info["initial_reward"]
    rewards.append(initial_reward)
    states.append(obs)

    while not done and not truncated:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(rewards[-1] + reward)
        states.append(obs)

    if truncated and not keep_unconverged:
        # If the episode was truncated, remove the last state and reward as they are not valid
        states.pop()
        rewards.pop()

    return rewards, states


def evaluate_airfoil_agent(agent, env, num_episodes=200, save_path=None):
    """
    Evaluates an RL agent on a set of airfoils from the UIUC airfoil database and prints a summary of the results.

    Args:
        agent: The RL agent to evaluate.
        env: The environment in which to run the episodes.
        num_episodes: The number of airfoils to evaluate. Default: 200.
        save_path: If provided, saves the results to a CSV file at this path. Default: ``None``.

    Returns:
        all_rewards: A list with the lists of cumulative rewards for each airfoil optimization.
        states: A list with lists of airfoil shapes for each step of the optimization of each airfoil.
    """
    all_rewards, states = [], []
    airfoil_list = random.sample(os.listdir(airfoil_database_root), num_episodes)
    if "utils" in airfoil_list:
        airfoil_list.remove("utils")
    for airfoil_name in airfoil_list:
        try:
            initial_airfoil = asb.Airfoil(airfoil_name)
        except:  # noqa: E722
            continue
        rewards, airfoils = run_episode(agent, env, initial_shape=initial_airfoil)
        all_rewards.append(rewards)
        states.append(airfoils)

    print_metric_summary(all_rewards, states,)
    if save_path is not None:
        save_results_to_csv(all_rewards, states, save_path)
    return all_rewards, states


def evaluate_airfoil_agent_whole_uiuc(agent, env, save_path=None):
    """
    Evaluates an RL agent on all airfoils in the UIUC airfoil database and prints a summary of the results.
    Args:
        agent: The RL agent to evaluate.
        env: The environment in which to run the episodes.
        save_path: If provided, saves the results to a CSV file at this path. Default: ``None``.
    Returns:
        all_rewards: A list with the lists of cumulative rewards for each airfoil optimization.
        states: A list with lists of airfoil shapes for each step of the optimization of each airfoil.
    """
    return evaluate_airfoil_agent(
        agent,
        env,
        num_episodes=len(os.listdir(airfoil_database_root)),
        save_path=save_path
    )

def evaluate_airfoil_agent_whole_uiuc_mpi(agent, env, save_results_path):
    """
    Evaluates an RL agent on all airfoils in the UIUC airfoil database using MPI for parallel processing and saves the results to a CSV file.
    If this function is used on a script, it should be run with `mpiexec -n <num_processes> python <script_name>.py`.

    Args:
        agent: The RL agent to evaluate.
        env: The environment in which to run the episodes.
        save_results_path: If provided, saves the results to a CSV file at this path.
    """
    rank = MPI_RANK
    size = MPI_SIZE

    active_workers = size - 1  # Number of workers currently active

    if rank == 0:
        # Master process
        airfoil_names = os.listdir(airfoil_database_root)
        if "utils" in airfoil_names:
            airfoil_names.remove("utils")
        results = []

        # Send initial work to each worker
        for i in range(1, size):
            if airfoil_names:
                airfoil_name = airfoil_names.pop()
                mpi_send(airfoil_name, dest=i, tag=1)

        # Receive results and distribute remaining work
        while active_workers > 0: #i < airfoil_count:
            status = MPI_Status()
            result = mpi_recv(source=MPI_ANY_SOURCE, tag=2, status=status)
            worker_rank = status.source
            if result[0] is not None and result[1] is not None:
                results.append(result[:2])

            if airfoil_names:
                airfoil_name = airfoil_names.pop()
                mpi_send(airfoil_name, dest=worker_rank, tag=1)
            else:
                mpi_send(None, dest=worker_rank, tag=0)  # Signal no more work
                active_workers -= 1

        # Finalize results
        rewards = [res[0] for res in results]
        states = [res[1] for res in results]
        print_metric_summary(rewards, states)

        if save_results_path is not None:
            save_results_to_csv(rewards, states, save_results_path)

    else:
        # Worker process
        while True:
            airfoil_name = mpi_recv(source=0, tag=MPI_ANY_TAG, status=MPI_Status())
            if airfoil_name is None:  # No more work
                break
            try:
                initial_airfoil = asb.Airfoil(airfoil_name)
                rewards, airfoils = run_episode(agent, env, initial_shape=initial_airfoil)
                data = (rewards, airfoils, airfoil_name)
            except Exception as e:
                pprint(rank, f"Error processing airfoil {airfoil_name}: {e}")
                data = (None, None, airfoil_name)  # Send None for rewards and airfoils if there's an error
            mpi_send(data, dest=0, tag=2)  # Send results back to master


def extract_metrics(rewards, states):
    initial_rewards = np.array([reward[0] for reward in rewards])
    final_rewards = np.array([reward[-1] for reward in rewards])
    best_rewards = np.array([max(reward) for reward in rewards])
    initial_states = [state[0] for state in states]
    best_states_idx = np.array([np.argmax(reward) for reward in rewards])
    best_states = [states[i][idx] for i, idx in enumerate(best_states_idx)]
    return initial_rewards, final_rewards, best_rewards, initial_states, best_states


def save_results_to_csv(rewards, states, save_results_path):
    initial_rewards, _, best_rewards, initial_states, best_states = extract_metrics(rewards, states)
    columns_names = ["initial_param_" + str(i) for i in range(initial_states[0].shape[0])]
    columns_names += ["best_param_" + str(i) for i in range(best_states[0].shape[0])]
    columns_names += ["initial_CLCD", "best_CLCD"]
    combined_data = np.column_stack((initial_states, best_states, initial_rewards, best_rewards))

    np.savetxt(save_results_path, combined_data, delimiter=',', 
               header=','.join(columns_names), comments='', fmt='%.6f')


def print_metric_summary(rewards, states):
    initial_rewards, _, best_rewards, _, _ = extract_metrics(rewards, states)
    # remove states that didn't converge
    converged_mask = (initial_rewards > NON_CONVERGED_REWARD) & (best_rewards > NON_CONVERGED_REWARD) & (initial_rewards != best_rewards)
    initial_rewards = initial_rewards[converged_mask]
    best_rewards = best_rewards[converged_mask]
    q75, q25 = np.percentile(best_rewards, [75 ,25])
    pprint(0, f"Number of airfoils converged: {len(best_rewards)}")
    pprint(0, f"Best CL/CD (median(IQR)): {round(np.median(best_rewards))} ({round(q75 - q25)})")
    pprint(0, f"Best CL/CD increment (mean +/- std): {(best_rewards - initial_rewards).mean():.1f}+/-{(best_rewards - initial_rewards).std():.1f}")
        