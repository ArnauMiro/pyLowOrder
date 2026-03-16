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
    hyperparams_serializer,
    get_optimizing_value,
    sample_params,
)

from .scalers import MinMaxScaler, StandardScaler, RobustScaler
from .schedulers import betaLinearScheduler
from .stats import RegressionEvaluator, ClassificationEvaluator
from .callbacks import EarlyStopper


# Backward-compatible activation wrappers (legacy import path: pyLOM.NN.utils)
def tanh():      return torch.nn.Tanh()
def relu():      return torch.nn.ReLU()
def elu():       return torch.nn.ELU()
def sigmoid():   return torch.nn.Sigmoid()
def leakyRelu(): return torch.nn.LeakyReLU()
def silu():      return torch.nn.SiLU()


def __getattr__(name):
    # Lazy legacy export to avoid circular imports with pyLOM.NN.dataset.
    if name == "Dataset":
        from ..dataset import Dataset
        return Dataset
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # experiment
    "plot_true_vs_pred",
    # optuna_utils
    "set_seed",
    "create_results_folder",
    "select_device",
    "count_trainable_params",
    "cleanup_tensors",
    "hyperparams_serializer",
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
    # activation wrappers
    "tanh",
    "relu",
    "elu",
    "sigmoid",
    "leakyRelu",
    "silu",
    # lazy legacy export
    "Dataset",
]
