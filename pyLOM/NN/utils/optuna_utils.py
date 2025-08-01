from typing import Union, Dict, Tuple, List
import os
import random
import gc
import numpy as np
import torch
from torch import Tensor

from .. import DEVICE

def set_seed(seed: int = 42, deterministic: bool = True, verbose: bool = False):
    """
    Sets the random seed across Python, NumPy, PyTorch and CUDA for reproducibility.

    Args:
        seed (int): The seed to use globally.
        deterministic (bool): Whether to force deterministic CuDNN operations (may affect performance).
        verbose (bool): Print seed setup info.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if verbose:
        print(f"[set_seed] Seed set to {seed} | Deterministic: {deterministic}")
 

def create_results_folder(RESUDIR: str, verbose: bool=True):
    r"""
    Create a folder to store the results of the neural network training.

    Args:
        RESUDIR (str): Path to the folder to be created.
        verbose (bool): If True, print messages to the console. Default is ``True``.
    """    
    if not os.path.exists(RESUDIR):
        os.makedirs(RESUDIR)
        if verbose: 
            print(f"Folder created: {RESUDIR}")
    elif verbose:
        print(f"Folder already exists: {RESUDIR}")
        

def select_device(device: str = DEVICE):
    r"""
    Select the device to be used for the training.

    Args:
        device (str): Device to be used. Default is cuda if available, otherwise cpu.
    """
    torch.device(device)
    return device


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model.
    Args:
        model (torch.nn.Module): The PyTorch model.
    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cleanup_tensors(tensors: Union[Tensor, Dict, None, Tuple, List]) -> None:
    if tensors is None:
        return

    # Eliminar referencias dentro de estructuras
    if isinstance(tensors, dict):
        for key in list(tensors.keys()):
            tensors[key] = None
    elif isinstance(tensors, (list, tuple)):
        for i in range(len(tensors)):
            tensors[i] = None
    else:
        tensors = None  # Tensor individual

    gc.collect()
    torch.cuda.empty_cache()

def hyperparams_serializer(obj) -> str:
        r"""
        Function used to print hyperparams in JSON format.
        Args:
            obj (Any): The object to serialize.
        Returns:
            str: The serialized object.
        """

        if hasattr(obj, "__class__"):  # Verify whether the object has a class
            return obj.__class__.__name__  # Return the class name
        raise TypeError(f"Type {type(obj)} not serializable")  # Raise an error if the object is not serializable

def get_optimizing_value(name: str, spec: Union[dict, list, int, float, str], trial) -> Union[int, float, str]:
    """
    Suggest a value for a given hyperparameter, supporting both modern dict-style
    and legacy list-style configuration formats.

    Args:
        name (str): Hyperparameter name.
        spec (Any): Can be:
            - dict with keys like {'type': 'int'/'float'/'categorical', ...}
            - list of [low, high] or categorical values
            - fixed value
        trial (optuna.Trial): Optuna trial object.

    Returns:
        Suggested value (int, float, str)
    """
    if isinstance(spec, dict):
        param_type = spec.get("type")
        if param_type in {"int", "float"}:
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"'low' and 'high' must be specified for '{param_type}' parameter '{name}'")

        if param_type == "int":
            return trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif param_type == "float":
            return trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step"),
                log=spec.get("log", False),
            )
        elif param_type == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported param type '{param_type}' in spec for '{name}'.")

    elif isinstance(spec, list):
        if all(isinstance(v, (int, float)) for v in spec) and len(spec) == 2:
            low, high = spec
            use_log = low > 0 and abs(high) / (abs(low) + 1e-8) >= 1000
            if isinstance(low, int) and isinstance(high, int):
                return trial.suggest_int(name, low, high, log=use_log)
            return trial.suggest_float(name, low, high, log=use_log)

        elif all(isinstance(v, str) for v in spec):
            return trial.suggest_categorical(name, spec)

        else:
            raise ValueError(
                f"List-style spec for '{name}' must be [low, high] for numeric or list of categories (str). Got: {spec}"
            )

    else:
        return spec  # Fixed value

