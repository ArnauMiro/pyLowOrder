"""
config_loader_factory.py — Configuration loader for pyLOM models (GNS, MLP, KAN)

This module provides utilities to parse model and training configurations
from YAML or dictionary-based inputs. It resolves string-based hyperparameters
(e.g., activation = "ReLU") into their corresponding torch classes.

Currently implemented:
    - GNS (Graph Neural Solver)

Placeholders exist for:
    - MLP
    - KAN

Author: Pablo Yeste
"""

from pathlib import Path
from typing import Union, Tuple

import yaml
import torch
from torch.nn import ELU, ReLU, LeakyReLU, Sigmoid, Tanh, PReLU, Softplus, GELU, SELU
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

from NN.utils.configs import GNSConfig, GNSTrainingConfig
# from NN.utils.configs import MLPConfig, MLPTrainingConfig
# from NN.utils.configs import KANConfig, KANTrainingConfig

from . import raiseError

# ─────────────────────────────────────────────────────
# Mapping dictionaries for common torch components
# ─────────────────────────────────────────────────────

ACTIVATIONS = {
    "ELU": ELU,
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
    "PReLU": PReLU,
    "Softplus": Softplus,
    "GELU": GELU,
    "SELU": SELU
}

LOSSES = {
    "MSELoss": torch.nn.MSELoss,
    "L1Loss": torch.nn.L1Loss,
    "SmoothL1Loss": torch.nn.SmoothL1Loss,
    "HuberLoss": torch.nn.HuberLoss
}

OPTIMIZERS = {
    "Adam": Adam,
    "SGD": SGD,
    "RMSprop": RMSprop,
    "AdamW": AdamW
}

SCHEDULERS = {
    "StepLR": StepLR,
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingLR": CosineAnnealingLR
}


# ─────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────

def load_yaml(path: Union[str, Path]) -> dict:
    """
    Load a YAML file and return its content as a dictionary.

    Args:
        path (Union[str, Path]): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_from_dict_or_torch(name: str, mapping: dict, module) -> Union[type, torch.nn.Module]:
    """
    Resolve a class from a string, prioritizing local mappings,
    and falling back to a PyTorch module (e.g. torch.nn or torch.optim).

    Allows extension with user-defined mappings beyond standard PyTorch classes.
    """

    if name in mapping:
        return mapping[name]
    if hasattr(module, name):
        return getattr(module, name)
    raiseError(f"Unknown name '{name}' not found in mapping or module '{module.__name__}'")

# ─────────────────────────────────────────────────────
# Model specific configuration loaders
# ─────────────────────────────────────────────────────

def load_gns_configs(config_dict: dict, with_training: bool = False) -> Union[GNSConfig, Tuple[GNSConfig, GNSTrainingConfig]]:
    """
    Parse and resolve GNS model configuration.

    Args:
        config_dict (dict): Parsed YAML configuration dictionary.
        with_training (bool): Whether to also parse the training configuration.

    Returns:
        GNSConfig or (GNSConfig, GNSTrainingConfig)
    """
    model_cfg = config_dict["model"].copy()
    exp_cfg = config_dict.get("experiment", {})

    model_cfg["device"] = exp_cfg.get("device", "cuda")
    model_cfg["seed"] = exp_cfg.get("seed", None)

    act_str = model_cfg.get("activation", "ELU")
    model_cfg["activation"] = _resolve_from_dict_or_torch(act_str, ACTIVATIONS, torch.nn)

    model_config = GNSConfig(**model_cfg)

    if not with_training:
        return model_config

    train_cfg = config_dict.get("training", {}).copy()
    optimizer = train_cfg.pop("optimizer", "Adam")
    scheduler = train_cfg.pop("scheduler", "StepLR")
    loss_fn = train_cfg.pop("loss_fn", "MSELoss")

    training_config = GNSTrainingConfig(
        **train_cfg,
        optimizer=_resolve_from_dict_or_torch(optimizer, OPTIMIZERS, torch.optim),
        scheduler=_resolve_from_dict_or_torch(scheduler, SCHEDULERS, torch.optim.lr_scheduler),
        loss_fn=_resolve_from_dict_or_torch(loss_fn, LOSSES, torch.nn)()
    )

    return model_config, training_config


def load_mlp_configs(config_dict: dict):
    """
    Placeholder for MLP configuration parsing.

    Args:
        config_dict (dict): The configuration dictionary.

    Raises:
        NotImplementedError
    """
    raiseError("MLP config loader not implemented yet.")

def load_kan_configs(config_dict: dict):
    """
    Placeholder for KAN configuration parsing.

    Args:
        config_dict (dict): The configuration dictionary.

    Raises:
        NotImplementedError
    """
    raiseError("KAN config loader not implemented yet.")

# ─────────────────────────────────────────────────────
# Dispatch map for model configuration loaders
# ─────────────────────────────────────────────────────

_LOADER_MAP = {
    "gns": load_gns_configs,
    "mlp": load_mlp_configs,
    "kan": load_kan_configs,
}

# ─────────────────────────────────────────────────────
# Entry points for external use
# ─────────────────────────────────────────────────────


def load_configs(
    config_path: Union[str, Path],
    model_type: str,
    with_training: bool = False
) -> Union[object, Tuple[object, object]]:
    """
    Load and resolve a full configuration from YAML, given the model type explicitly.

    Args:
        config_path (str or Path): Path to YAML file.
        model_type (str): Type of model to load ("gns", "mlp", "kan").
        with_training (bool): If True, returns (model_config, training_config). Else only model_config.

    Returns:
        ModelConfig or (ModelConfig, TrainingConfig)
    """
    config_dict = load_yaml(config_path)
    model_type = model_type.lower()

    loader_fn = _LOADER_MAP.get(model_type)
    if loader_fn is None:
        raiseError(f"Unsupported model_type '{model_type}'. Must be one of {list(_LOADER_MAP)}.")

    return loader_fn(config_dict, with_training=with_training)


def _model_from_config_path(
    model_class,
    yaml_path: Union[str, Path],
    model_type: str,
    with_training: bool = False
):
    """
    Instantiate a model from a YAML configuration file, given the model type explicitly.

    Args:
        model_class: Model class to instantiate (must accept config as first arg).
        yaml_path (str or Path): Path to YAML file.
        model_type (str): Type of model ("gns", "mlp", "kan").
        with_training (bool): Whether to return training config as well.

    Returns:
        model instance or (model, training_config)
    """
    model_type = model_type.lower()
    loader_fn = _LOADER_MAP.get(model_type)
    if loader_fn is None:
        raiseError(f"Unsupported model_type '{model_type}'. Must be one of {list(_LOADER_MAP)}.")

    config_dict = load_yaml(yaml_path)
    configs = loader_fn(config_dict, with_training=with_training)

    if with_training:
        model_config, training_config = configs
        model = model_class(config=model_config)
        return model, training_config
    else:
        model_config = configs
        model = model_class(config=model_config)
        return model


