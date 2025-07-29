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
from typing import Union, Tuple, Dict

import yaml
import torch
from torch.nn import ELU, ReLU, LeakyReLU, Sigmoid, Tanh, PReLU, Softplus, GELU, SELU
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

from .configs import GNSConfig, GNSTrainingConfig
# from ..NN.utils.configs import MLPConfig, MLPTrainingConfig
# from ..NN.utils.configs import KANConfig, KANTrainingConfig

from ...utils import raiseError

__all__ = [
    "load_configs",
    "load_yaml",
    "_model_from_config_path",
    "_resolve_optuna_trial_params",
]

# ─────────────────────────────────────────────────────
# Mapping dictionaries for common torch components
# ─────────────────────────────────────────────────────

ACTIVATIONS = {
    "ELU": ELU, "ReLU": ReLU, "LeakyReLU": LeakyReLU,
    "Sigmoid": Sigmoid, "Tanh": Tanh, "PReLU": PReLU,
    "Softplus": Softplus, "GELU": GELU, "SELU": SELU
}

LOSSES = {
    "MSELoss": torch.nn.MSELoss,
    "L1Loss": torch.nn.L1Loss,
    "SmoothL1Loss": torch.nn.SmoothL1Loss,
    "HuberLoss": torch.nn.HuberLoss
}

OPTIMIZERS = {
    "Adam": Adam, "SGD": SGD, "RMSprop": RMSprop, "AdamW": AdamW
}

SCHEDULERS = {
    "StepLR": StepLR, "ExponentialLR": ExponentialLR, "CosineAnnealingLR": CosineAnnealingLR
}

_ANNOTATIONS_MAP = {
    "gns": ("GNSConfig", "GNSTrainingConfig"),
    "mlp": ("MLPConfig", "MLPTrainingConfig"),
    "kan": ("KANConfig", "KANTrainingConfig"),
}

# ─────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────

def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _resolve_from_dict_or_torch(name: str, mapping: dict, module) -> Union[type, torch.nn.Module]:
    if name in mapping:
        return mapping[name]
    if hasattr(module, name):
        return getattr(module, name)
    raiseError(f"Unknown name '{name}' not found in mapping or module '{module.__name__}'.")

# ─────────────────────────────────────────────────────
# Generic config resolution helper
# ─────────────────────────────────────────────────────

def _resolve_model_and_training_configs(
    model_cfg_raw: dict,
    train_cfg_raw: dict,
    exp_cfg: dict,
    model_cls,
    training_cls,
) -> Tuple[object, object]:
    model_cfg = model_cfg_raw.copy()
    train_cfg = train_cfg_raw.copy()

    model_cfg["device"] = exp_cfg.get("device", "cuda")
    model_cfg["seed"] = exp_cfg.get("seed", None)
    if "activation" in model_cfg:
        model_cfg["activation"] = _resolve_from_dict_or_torch(model_cfg["activation"], ACTIVATIONS, torch.nn)

    optimizer = train_cfg.pop("optimizer", "Adam")
    scheduler = train_cfg.pop("scheduler", "StepLR")
    loss_fn = train_cfg.pop("loss_fn", "MSELoss")

    model_config = model_cls(**model_cfg)
    training_config = training_cls(
        **train_cfg,
        optimizer=_resolve_from_dict_or_torch(optimizer, OPTIMIZERS, torch.optim),
        scheduler=_resolve_from_dict_or_torch(scheduler, SCHEDULERS, torch.optim.lr_scheduler),
        loss_fn=_resolve_from_dict_or_torch(loss_fn, LOSSES, torch.nn)(),
    )
    return model_config, training_config

# ─────────────────────────────────────────────────────
# Model specific configuration loaders
# ─────────────────────────────────────────────────────

def load_gns_configs(config_dict: dict, with_training: bool = False):
    exp_cfg = config_dict.get("experiment", {})
    model_cfg = config_dict["model"]
    if not with_training:
        model_cfg["device"] = exp_cfg.get("device", "cuda")
        model_cfg["seed"] = exp_cfg.get("seed", None)
        model_cfg["activation"] = _resolve_from_dict_or_torch(model_cfg["activation"], ACTIVATIONS, torch.nn)
        return GNSConfig(**model_cfg)

    return _resolve_model_and_training_configs(
        model_cfg_raw=config_dict["model"],
        train_cfg_raw=config_dict["training"],
        exp_cfg=exp_cfg,
        model_cls=GNSConfig,
        training_cls=GNSTrainingConfig,
    )

def load_mlp_configs(config_dict: dict):
    raiseError("MLP config loader not implemented yet.")

def load_kan_configs(config_dict: dict):
    raiseError("KAN config loader not implemented yet.")

# ─────────────────────────────────────────────────────
# Dispatch and interface
# ─────────────────────────────────────────────────────

_LOADER_MAP = {
    "gns": load_gns_configs,
    "mlp": load_mlp_configs,
    "kan": load_kan_configs,
}

def load_configs(
    config_path: Union[str, Path],
    model_type: str,
    with_training: bool = False
) -> Union[object, Tuple[object, object]]:
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
    model_type = model_type.lower()
    loader_fn = _LOADER_MAP.get(model_type)
    if loader_fn is None:
        raiseError(f"Unsupported model_type '{model_type}'. Must be one of {list(_LOADER_MAP)}.")
    config_dict = load_yaml(yaml_path)
    configs = loader_fn(config_dict, with_training=with_training)
    if with_training:
        model_config, training_config = configs
        return model_class(config=model_config), training_config
    return model_class(config=configs)

def _resolve_optuna_trial_params(
    hyperparams: Dict,
    model_type: str
) -> Tuple[object, object]:
    from NN.utils import configs as config_module

    model_type = model_type.lower()
    try:
        model_cls_name, train_cls_name = _ANNOTATIONS_MAP[model_type]
        model_cls = getattr(config_module, model_cls_name)
        train_cls = getattr(config_module, train_cls_name)
    except KeyError:
        raiseError(f"No config annotations found for model_type '{model_type}'.")

    model_keys = set(model_cls.__annotations__)
    training_keys = set(train_cls.__annotations__)

    model_cfg = {k: v for k, v in hyperparams.items() if k in model_keys}
    training_cfg = {k: v for k, v in hyperparams.items() if k in training_keys}
    experiment_cfg = {k: v for k in ["seed", "device"] if k in hyperparams}

    return _resolve_model_and_training_configs(
        model_cfg_raw=model_cfg,
        train_cfg_raw=training_cfg,
        exp_cfg=experiment_cfg,
        model_cls=model_cls,
        training_cls=train_cls
    )
