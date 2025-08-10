from typing import Union, Dict, Tuple, List, Any
import os
import random
import gc
import numpy as np
import torch
from torch import Tensor

from .. import DEVICE

def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility.
    Args:
        seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Backwards compatible for older PyTorch versions
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _worker_init_fn(_):
    """Seed Python & NumPy per worker, derived from PyTorch worker seed.

    Notes
    -----
    - torch.initial_seed() already incorporates rank/worker id.
    - Use % 2**32 to obtain a numpy valid integer.
    - Do not use torch.manual_seed() here: Pytorch already sets each worker internally.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
 

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

def get_optimizing_value(name: str, spec: Any, trial):
    """Suggest a single hyperparameter value from a spec."""
    if not isinstance(spec, dict) or "type" not in spec:
        return spec  # fixed value

    ptype = spec["type"]

    if ptype == "int":
        return trial.suggest_int(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step", 1),
            log=spec.get("log", False),
        )
    elif ptype == "float":
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step"),
            log=spec.get("log", False),
        )
    elif ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"Unsupported param type '{ptype}' for '{name}'")


def sample_params(space: dict, trial, prefix="") -> dict:
    """
    Recursively sample an Optuna search space dict.

    Args:
        space: Nested dict with fixed values or {type: ...} specs.
        trial: Optuna trial.
        prefix: Name prefix for hierarchical parameters.
    """
    result = {}
    for k, v in space.items():
        full_name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and "type" not in v:
            # Nested section → recurse
            result[k] = sample_params(v, trial, prefix=full_name)
        else:
            result[k] = get_optimizing_value(full_name, v, trial)
    return result


