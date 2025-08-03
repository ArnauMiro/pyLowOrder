from typing import Dict, Union
from pathlib import Path
import yaml

import torch
from torch.nn import ELU, ReLU, LeakyReLU, Sigmoid, Tanh, PReLU, Softplus, GELU, SELU
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from optuna.pruners import MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler

# ─────────────────────────────────────────────────────
# Torch mappings
# ─────────────────────────────────────────────────────

_MAPPING_KEYS: Dict[str, Dict] = {
    "activation": {
        "ELU": ELU, "ReLU": ReLU, "LeakyReLU": LeakyReLU,
        "Sigmoid": Sigmoid, "Tanh": Tanh, "PReLU": PReLU,
        "Softplus": Softplus, "GELU": GELU, "SELU": SELU
    },
    "loss_fn": {
        "MSELoss": torch.nn.MSELoss,
        "L1Loss": torch.nn.L1Loss,
        "SmoothL1Loss": torch.nn.SmoothL1Loss,
        "HuberLoss": torch.nn.HuberLoss
    },
    "optimizer": {
        "Adam": Adam, "SGD": SGD, "RMSprop": RMSprop, "AdamW": AdamW
    },
    "scheduler": {
        "StepLR": StepLR, "ExponentialLR": ExponentialLR, "CosineAnnealingLR": CosineAnnealingLR
    },
    "sampler": {
        "TPESampler": TPESampler,
        "RandomSampler": RandomSampler,
        "CmaEsSampler": CmaEsSampler
    },
    "pruner": {
        "MedianPruner": MedianPruner,
        "NopPruner": NopPruner,
        "SuccessiveHalvingPruner": SuccessiveHalvingPruner
    }
}



def serialize_config(cfg: dict) -> dict:
    """Serialize a configuration dictionary to a format suitable for storage.
    Args:
        cfg (dict): Configuration dictionary with keys that may include torch or optuna objects.
    Returns:
        dict: Serialized configuration dictionary with string representations of objects.
    """
    serialized = {}
    for key, value in cfg.items():
        if key in _MAPPING_KEYS:
            reverse = {}
            for k, v in _MAPPING_KEYS[key].items():
                reverse[v] = k
                if isinstance(v, type):
                    reverse[v()] = k  # Add instance too

            if isinstance(value, type):
                key_name = reverse.get(value)
            else:
                key_name = reverse.get(value.__class__, value.__class__.__name__)
            if key_name is None:
                raise ValueError(f"Cannot serialize value '{value}' for key '{key}'")
            serialized[key] = key_name
        else:
            serialized[key] = value
    return serialized




def deserialize_config(cfg: dict) -> dict:
    """Deserialize a configuration dictionary from a stored format.
    Args:
        cfg (dict): Configuration dictionary with string representations of objects.
    Returns:
        dict: Deserialized configuration dictionary with actual torch or optuna objects.
    """
    deserialized = {}
    for key, value in cfg.items():
        if key in _MAPPING_KEYS:
            mapping = _MAPPING_KEYS[key]
            if isinstance(value, str):
                if value not in mapping:
                    raise ValueError(f"Unknown value '{value}' for key '{key}'")
                resolved = mapping[value]
                deserialized[key] = resolved() if callable(resolved) and not isinstance(resolved, type) else resolved
            else:
                deserialized[key] = value
        else:
            deserialized[key] = value
    return deserialized


def load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML file and return its contents as a dictionary.
    Args:
        path (Union[str, Path]): Path to the YAML file.
    Returns:
        dict: Contents of the YAML file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
