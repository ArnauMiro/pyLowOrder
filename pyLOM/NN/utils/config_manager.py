from pathlib import Path
from typing import Union, Optional, Dict, Type
from dataclasses import asdict

import yaml
import torch
from torch.nn import ELU, ReLU, LeakyReLU, Sigmoid, Tanh, PReLU, Softplus, GELU, SELU
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

from .configs import GNSModelConfig, GNSFitConfig
from ..optimizer import OptunaOptimizer
from ...utils import raiseError

# ─────────────────────────────────────────────────────
# Torch mappings
# ─────────────────────────────────────────────────────

_ACTIVATIONS = {
    "ELU": ELU, "ReLU": ReLU, "LeakyReLU": LeakyReLU,
    "Sigmoid": Sigmoid, "Tanh": Tanh, "PReLU": PReLU,
    "Softplus": Softplus, "GELU": GELU, "SELU": SELU
}

_LOSSES = {
    "MSELoss": torch.nn.MSELoss,
    "L1Loss": torch.nn.L1Loss,
    "SmoothL1Loss": torch.nn.SmoothL1Loss,
    "HuberLoss": torch.nn.HuberLoss
}

_OPTIMIZERS = {
    "Adam": Adam, "SGD": SGD, "RMSprop": RMSprop, "AdamW": AdamW
}

_SCHEDULERS = {
    "StepLR": StepLR, "ExponentialLR": ExponentialLR, "CosineAnnealingLR": CosineAnnealingLR
}

# ─────────────────────────────────────────────────────
# Config resolver base
# ─────────────────────────────────────────────────────

class BaseModelConfigResolver:
    def __init__(self, raw: dict, experiment: dict):
        self.raw = raw
        self.experiment = experiment

    def resolve_model_config(self):
        raise NotImplementedError

    def resolve_fit_config(self):
        raise NotImplementedError

    def resolve_optuna_config(self):
        raise NotImplementedError

# ─────────────────────────────────────────────────────
# GNS Config
# ─────────────────────────────────────────────────────

class GNSConfig(BaseModelConfigResolver):
    def __init__(self, config: Union[dict, str, Path]) -> None:
        if isinstance(config, (str, Path)):
            raw = self._load_yaml(config)
        elif isinstance(config, dict):
            raw = config
        else:
            raise TypeError("Expected dict or YAML path as configuration input.")

        self.raw = raw
        self.experiment = raw.get("experiment", {})
        super().__init__(raw, self.experiment)

        self.model: Optional[GNSModelConfig] = None
        self.fit: Optional[GNSFitConfig] = None
        self.optuna: Optional[Dict[str, Union[dict, OptunaOptimizer]]] = None

    def resolve(self) -> None:
        if "model" in self.raw:
            self.model = self.resolve_model_config()
        if "fit" in self.raw:
            self.fit = self.resolve_fit_config()
        if "optuna" in self.raw:
            self.optuna = self.resolve_optuna_config()

        if not any([self.model, self.fit, self.optuna]):
            raiseError("Invalid configuration: no valid sections found (expected 'model', 'fit' or 'optuna').")

    def resolve_model_config(self) -> GNSModelConfig:
        model_cfg = self.raw["model"].copy()

        # Inject experiment seed
        model_cfg["seed"] = self.experiment.get("seed", 1234)
        model_cfg["device"] = torch.device(self.experiment.get("device", "cuda"))

        if "activation" in model_cfg:
            resolved = self._resolve_from_dict_or_torch(model_cfg["activation"], _ACTIVATIONS, torch.nn)
            model_cfg["activation"] = resolved() if isinstance(resolved, type) else resolved

        return GNSModelConfig(**model_cfg)

    def resolve_fit_config(self) -> GNSFitConfig:
        train_cfg = self.raw["fit"].copy()

        optimizer = train_cfg.pop("optimizer", "Adam")
        scheduler = train_cfg.pop("scheduler", "StepLR")
        loss_fn = train_cfg.pop("loss_fn", "MSELoss")

        return GNSFitConfig(
            **train_cfg,
            optimizer=self._resolve_from_dict_or_torch(optimizer, _OPTIMIZERS, torch.optim),
            scheduler=self._resolve_from_dict_or_torch(scheduler, _SCHEDULERS, torch.optim.lr_scheduler),
            loss_fn=self._resolve_from_dict_or_torch(loss_fn, _LOSSES, torch.nn)(),
        )

    def resolve_optuna_config(self) -> OptunaOptimizer:
        raw = self.raw["optuna"]
        study_params = raw.get("study", {})
        optimization_params = raw.get("optimization_params", {}).copy()

        # Inject seed from experiment
        seed = self.experiment.get("seed", 1234)

        graph_path = raw.get("graph_path")
        if not graph_path:
            raiseError("Missing 'graph_path' in 'optuna.graph_path'.")

        return {
            'optimization_params': optimization_params,
            'sampler': None,
            'pruner': study_params.get("pruner"),
            'direction': study_params.get("direction", "minimize"),
            'n_trials': study_params.get("n_trials", 100),
            'save_dir': study_params.get("save_dir"),
            'seed': seed,
            'graph_path': graph_path,
        }

    def _resolve_from_dict_or_torch(self, name: str, mapping: dict, module) -> Union[type, torch.nn.Module]:
        if name in mapping:
            return mapping[name]
        if hasattr(module, name):
            return getattr(module, name)
        raiseError(f"Unknown name '{name}' not found in mapping or module '{module.__name__}'.")

    @staticmethod
    def _load_yaml(path: Union[str, Path]) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def to_dict(self) -> dict:
        return {
            "model": asdict(self.model) if self.model else None,
            "fit": asdict(self.fit) if self.fit else None,
            "experiment": self.experiment,
            "optuna": {
                k: v if k != "optuna_optimizer" else None
                for k, v in self.optuna.items()
            } if self.optuna else None,
        }

    def __repr__(self) -> str:
        name = self.experiment.get("name", "Unnamed")
        version = self.experiment.get("version", "?")
        return f"<GNSConfig: experiment='{name}', version={version}>"

