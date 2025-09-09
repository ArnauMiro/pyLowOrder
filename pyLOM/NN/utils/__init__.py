"""
pyLOM.NN.utils
----------------
Public API for neural-network utilities in pyLOM.

This module re-exports commonly used utilities for convenience.
Keep imports lightweight; avoid importing heavy third-party libs here.
"""

from .experiment import (
    evaluate_model,
    compute_regression_metrics,
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

from .scalers import MinMaxScaler
from .schedulers import betaLinearScheduler
from .stats import RegressionEvaluator
from .callbacks import EarlyStopper


__all__ = [
    # experiment
    "evaluate_model",
    "compute_regression_metrics",
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
    "betaLinearScheduler",
    "RegressionEvaluator",
    "EarlyStopper",
]
