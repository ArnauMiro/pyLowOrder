import os
import random

import aerosandbox as asb
import numpy as np
import pandas as pd
from aerosandbox import KulfanAirfoil, _asb_root
from tqdm import tqdm

from pyLOM.utils import pprint

airfoil_database_root = _asb_root / "geometry" / "airfoil" / "airfoil_database"

def run_episode(rl_model, env, reset_options={}, seed=None, keep_unconverged=False):
    done = False
    truncated = False
    states = []
    rewards = []
    
    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        env.action_space.seed(seed)
    
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
    all_rewards, states = [], []
    airfoil_list = random.sample(os.listdir(airfoil_database_root), num_episodes)
    if "utils" in airfoil_list:
        airfoil_list.remove("utils")
    pbar = tqdm(airfoil_list, desc="Evaluating airfoils", unit="airfoil")
    for airfoil_name in pbar:
        try:
            initial_airfoil = asb.Airfoil(airfoil_name)
        except:  # noqa: E722
            continue
        rewards, airfoils = run_episode(agent, env, reset_options={'initial_airfoil': initial_airfoil})
        all_rewards.append(rewards)
        states.append(airfoils)

    if save_path is not None:
        save_results_to_csv(all_rewards, states, save_path)
    return all_rewards, states


def evaluate_airfoil_agent_whole_uiuc(agent, env, save_path=None):
    return evaluate_airfoil_agent(
        agent,
        env,
        num_episodes=len(os.listdir(airfoil_database_root)),
        save_path=save_path
    )

def evaluate_airfoil_agent_whole_uiuc_mpi(agent, env, save_results_path=None):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    active_workers = size - 1  # Number of workers currently active

    if rank == 0:
        # Master process
        airfoil_names = os.listdir(airfoil_database_root)
        if "utils" in airfoil_names:
            airfoil_names.remove("utils")
        airfoil_count = len(airfoil_names)
        results = []

        # Send initial work to each worker
        for i in range(1, size):
            if airfoil_names:
                airfoil_name = airfoil_names.pop()
                comm.send(airfoil_name, dest=i, tag=1)

        # Receive results and distribute remaining work
        with tqdm(total=airfoil_count, desc="Evaluating airfoils", unit="airfoil") as pbar:
            while active_workers > 0: #i < airfoil_count:
                status = MPI.Status()
                result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
                worker_rank = status.source
                if result[0] is not None and result[1] is not None:
                    results.append(result[:2])

                pbar.update(1)
                if airfoil_names:
                    airfoil_name = airfoil_names.pop()
                    comm.send(airfoil_name, dest=worker_rank, tag=1)
                else:
                    comm.send(None, dest=worker_rank, tag=0)  # Signal no more work
                    active_workers -= 1

        # Finalize results
        rewards = [res[0] for res in results]
        states = [res[1] for res in results]

        if save_results_path is not None:
            save_results_to_csv(rewards, states, save_results_path)

    else:
        # Worker process
        while True:
            airfoil_name = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
            if airfoil_name is None:  # No more work
                break
            try:
                initial_airfoil = asb.Airfoil(airfoil_name)
                rewards, airfoils = run_episode(agent, env, reset_options={'initial_airfoil': initial_airfoil})
                comm.send((rewards, airfoils, airfoil_name), dest=0, tag=2)
            except Exception as e:
                pprint(rank, f"Error processing airfoil {airfoil_name}: {e}")
                comm.send((None, None, None), dest=0, tag=2)


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
    df = pd.DataFrame(np.hstack((initial_states, best_states)), columns=columns_names)
    df["initial_CLCD"] = initial_rewards
    df["best_CLCD"] = best_rewards
    df.to_csv(save_results_path, index=False)


