"""
pyLOM.NN.utils
----------------
Public API for neural-network utilities in pyLOM.

This module re-exports commonly used utilities for convenience.
Keep imports lightweight; avoid importing heavy third-party libs here.
"""

import torch

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

from .config_resolvers import load_yaml, dataclass_from_dict, to_native, instantiate_from_config
from .config_resolvers import resolve_import, resolve_device, resolve_activation, resolve_loss, resolve_optimizer, resolve_scheduler
from .config_schema    import GNSModelConfig, GNSTrainingConfig
from .experiment       import plot_true_vs_pred, save_experiment_artifacts, plot_train_test_loss
from .optuna_utils     import set_seed, create_results_folder, select_device, count_trainable_params, cleanup_tensors, get_optimizing_value, sample_params
from .scalers          import MinMaxScaler, StandardScaler, RobustScaler
from .schedulers       import betaLinearScheduler
from .stats            import RegressionEvaluator, ClassificationEvaluator
from .callbacks        import EarlyStopper


del torch, scalers, schedulers, stats, callbacks, optuna_utils, experiment
