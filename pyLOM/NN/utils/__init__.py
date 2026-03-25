"""
pyLOM.NN.utils
----------------
Public API for neural-network utilities in pyLOM.

This module re-exports commonly used utilities for convenience.
Keep imports lightweight; avoid importing heavy third-party libs here.
"""

import torch

from .experiment import (
    plot_true_vs_pred,
)

from .optuna_utils import (
    set_seed,
    create_results_folder,
    select_device,
    count_trainable_params,
    cleanup_tensors,
    get_optimizing_value,
    sample_params,
)

from .scalers import MinMaxScaler, StandardScaler, RobustScaler
from .schedulers import betaLinearScheduler
from .stats import RegressionEvaluator, ClassificationEvaluator
from .callbacks import EarlyStopper


# Activation constructors shared across NN modules
def tanh() -> torch.nn.Module:
    return torch.nn.Tanh()

def relu() -> torch.nn.Module:
    return torch.nn.ReLU()

def elu() -> torch.nn.Module:
    return torch.nn.ELU()

def sigmoid() -> torch.nn.Module:
    return torch.nn.Sigmoid()

def leakyRelu() -> torch.nn.Module:
    return torch.nn.LeakyReLU()

def silu() -> torch.nn.Module:
    return torch.nn.SiLU()





__all__ = [
    # experiment
    "plot_true_vs_pred",
    # optuna_utils
    "set_seed",
    "create_results_folder",
    "select_device",
    "count_trainable_params",
    "cleanup_tensors",
    "get_optimizing_value",
    "sample_params",
    # scalers, schedulers, metrics, callbacks
    "MinMaxScaler",
    "StandardScaler",
    "RobustScaler",
    "betaLinearScheduler",
    "RegressionEvaluator",
    "ClassificationEvaluator",
    "EarlyStopper",
    "leakyRelu",
    "sigmoid",
    "elu",
    "relu",
    "tanh",
    "silu",
]
