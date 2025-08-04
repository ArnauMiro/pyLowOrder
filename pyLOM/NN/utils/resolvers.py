from typing import Any, Dict
import torch

# Predefined mappings for PyTorch and Optuna objects
_MAPPING_KEYS: Dict[str, Dict[str, Any]] = {
    "activation": {
        "ELU": torch.nn.ELU,
        "ReLU": torch.nn.ReLU,
        "LeakyReLU": torch.nn.LeakyReLU,
        "Sigmoid": torch.nn.Sigmoid,
        "Tanh": torch.nn.Tanh,
        "PReLU": torch.nn.PReLU,
        "Softplus": torch.nn.Softplus,
        "GELU": torch.nn.GELU,
        "SELU": torch.nn.SELU
    },
    "loss_fn": {
        "MSELoss": torch.nn.MSELoss,
        "L1Loss": torch.nn.L1Loss,
        "SmoothL1Loss": torch.nn.SmoothL1Loss,
        "HuberLoss": torch.nn.HuberLoss
    },
    "optimizer": {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop,
        "AdamW": torch.optim.AdamW
    },
    "scheduler": {
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR
    },
    "sampler": {
        "TPESampler": "optuna.samplers.TPESampler",
        "RandomSampler": "optuna.samplers.RandomSampler",
        "CmaEsSampler": "optuna.samplers.CmaEsSampler"
    },
    "pruner": {
        "MedianPruner": "optuna.pruners.MedianPruner",
        "NopPruner": "optuna.pruners.NopPruner",
        "SuccessiveHalvingPruner": "optuna.pruners.SuccessiveHalvingPruner"
    }
}
