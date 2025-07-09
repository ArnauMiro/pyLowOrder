from typing import Union, Dict, Tuple, List
import os
import random
import gc
import numpy as np
import torch
from torch import Tensor

from .. import DEVICE

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
 
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
 

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

def get_optimizing_value(name, value, trial) -> Union[int, float, str]:
        """
        Suggest a value for a given hyperparameter depending on its type and content.

        Args:
            name (str): Hyperparameter name.
            value (Any): Either a fixed value, a range, or a list of options.
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Union[int, float, str]: Suggested value.
        """
        if isinstance(value, (tuple, list)):
            if all(isinstance(v, (int, float)) for v in value):
                use_log = np.abs(value[1]) / (np.abs(value[0]) + 1e-8) >= 1000
                low, high = value

                if isinstance(low, int):
                    if name == "latent_dim":
                        return trial.suggest_int(name, low + low % 2, high + high % 2, step=2)
                    return trial.suggest_int(name, low, high, log=use_log)

                elif isinstance(low, float):
                    if use_log and low <= 0:
                        use_log = False  # fallback to linear scale
                    return trial.suggest_float(name, low, high, log=use_log)
            elif all(isinstance(v, str) for v in value):
                return trial.suggest_categorical(name, value)
            else:
                raise ValueError(f"Unsupported value list for {name}: {value}")
        else:
            return value