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
    """
    Serialize a configuration dictionary by converting object instances to string identifiers
    using predefined mappings.
    """
    serialized = {}
    for key, value in cfg.items():
        if key in _MAPPING_KEYS:
            reverse = {v: k for k, v in _MAPPING_KEYS[key].items()}

            # Special case: if value is an instance of mapped type
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                cls = value.__class__
                if cls in reverse:
                    serialized[key] = reverse[cls]
                else:
                    raise ValueError(f"Cannot serialize instance of {cls} for key '{key}'")

            # If value is already a class
            elif isinstance(value, type) and value in reverse:
                serialized[key] = reverse[value]

            else:
                raise ValueError(f"Cannot serialize value '{value}' for key '{key}'")

        else:
            serialized[key] = value
    return serialized


def deserialize_config(cfg: dict) -> dict:
    """
    Deserialize a configuration dictionary by converting string identifiers to object instances
    using predefined mappings.
    """
    deserialized = {}
    for key, value in cfg.items():
        if key in _MAPPING_KEYS:
            mapping = _MAPPING_KEYS[key]
            if isinstance(value, str):
                if value not in mapping:
                    raise ValueError(f"Unknown value '{value}' for key '{key}'")
                cls = mapping[value]
                # Don't instantiate optimizers or schedulers yet
                if key in ("optimizer", "scheduler"):
                    deserialized[key] = cls  # Leave as class for later instantiation
                else:
                    deserialized[key] = cls()  # Instantiate activation, loss, etc.
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
