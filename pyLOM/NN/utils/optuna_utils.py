from typing import Union, Dict, Tuple, List, Any
import os
import random
import gc
import numpy as np
import torch
from torch import Tensor

from .. import DEVICE


import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for full reproducibility across Python, NumPy, and PyTorch.
    
    This function enforces determinism in all major sources of randomness:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - cuBLAS/cuDNN kernels
    
    It also sets relevant environment variables to avoid hidden sources
    of non-determinism (e.g., Python hash seed, cuBLAS workspace).
    
    Note
    ----
    - Setting `torch.use_deterministic_algorithms(True)` may raise an error 
      if your code calls non-deterministic ops. In that case, rewrite those ops 
      or handle them explicitly.
    - Some performance optimizations (e.g., `torch.backends.cudnn.benchmark`) 
      are disabled to guarantee determinism.
    
    Args
    ----
    seed : int, optional
        The seed value to use for all RNGs (default = 42).
    """
    # -------------------------------
    # Python & NumPy RNGs
    # -------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)  # enforce deterministic hashing in Python
    random.seed(seed)
    np.random.seed(seed)

    # -------------------------------
    # PyTorch CPU & CUDA RNGs
    # -------------------------------
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # -------------------------------
    # cuBLAS/cuDNN Determinism
    # -------------------------------
    # cuBLAS (matrix multiplications) may introduce nondeterminism unless configured
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # or ":16:8" if OOM occurs

    try:
        # Enforce deterministic algorithms globally (PyTorch >= 1.8)
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        # Fallback for older versions
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -------------------------------
    # Disable TF32 for Ampere+ GPUs
    # -------------------------------
    # TF32 can cause slight numerical differences across runs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def set_seed_legacy(seed: int = 42) -> None:
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

def _is_dist_spec(x: object) -> bool:
    return isinstance(x, dict) and "type" in x

def _materialize_space(space: dict, flat: dict, prefix: str = "") -> dict:
    """
    Reemplaza specs {type:...} en `space` por valores de `flat` (claves punteadas).
    Mantiene constantes y recorre recursivamente dicts anidados.
    """
    out = {}
    for k, v in space.items():
        path = f"{prefix}.{k}" if prefix else k
        if _is_dist_spec(v):
            if path not in flat:
                raise ValueError(f"Missing sampled value for '{path}'")
            out[k] = flat[path]
        elif isinstance(v, dict):
            out[k] = _materialize_space(v, flat, path)
        else:
            out[k] = flat.get(path, v)
    return out



