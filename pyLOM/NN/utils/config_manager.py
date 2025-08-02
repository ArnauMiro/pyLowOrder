# config_manager.py

from pathlib import Path
from typing import Union, Optional, Dict, Type
from dataclasses import asdict

import yaml
import torch
from torch.nn import ELU, ReLU, LeakyReLU, Sigmoid, Tanh, PReLU, Softplus, GELU, SELU
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

from optuna.pruners import MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler

from .dataclasses import GNSParams, GNSTraining
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

_SAMPLERS = {
    "TPESampler": TPESampler,
    "RandomSampler": RandomSampler,
    "CmaEsSampler": CmaEsSampler
}

_PRUNERS = {
    "MedianPruner": MedianPruner,
    "NopPruner": NopPruner,
    "SuccessiveHalvingPruner": SuccessiveHalvingPruner
}

# ─────────────────────────────────────────────────────
# Config resolver base
# ─────────────────────────────────────────────────────

class BaseModelConfigResolver:
    def __init__(self, raw: dict):
        self.raw = raw

    def resolve_model_config(self):
        ...

    def resolve_training_config(self):
        ...

    def resolve_optuna_dict(self):
        ...

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
        super().__init__(raw)

        self.graph_path = None
        self.model_cfg: Optional[GNSParams] = None
        self.training_cfg: Optional[GNSTraining] = None
        self.optuna_dict: Optional[Dict] = None

    def resolve(self) -> None:
        if "model" in self.raw:
            self.graph_path = self.raw["model"].get("graph_path")
            model_params = self.raw["model"].get("params")
            self.model_cfg = self.resolve_model_config(model_params)
        if "training" in self.raw:
            training_params = self.raw["training"]
            self.training_cfg = self.resolve_training_config(training_params)
        if "optuna" in self.raw:
            optuna_params = self.raw["optuna"]
            self.optuna_dict = self.resolve_optuna_dict(optuna_params)

        if not any([self.model_cfg, self.training_cfg, self.optuna_dict]):
            raiseError("Invalid configuration: no valid sections found (expected 'model', 'training' or 'optuna').")

    def resolve_model_config(self, model_params: Dict) -> GNSParams:
        raw_model = model_params.copy()
        model_params = raw_model.get("params").copy()

        # Resolve activation
        if "activation" in model_params:
            resolved = self._resolve_from_dict_or_torch(model_params["activation"], _ACTIVATIONS, torch.nn)
            model_params["activation"] = resolved() if isinstance(resolved, type) else resolved

        # Resolve device
        if "device" in model_params:
            model_params["device"] = torch.device(model_params["device"])

        return GNSParams(**model_params)

    def resolve_training_config(self, training_params: Dict) -> GNSTraining:
        train_cfg = training_params.copy()
        optimizer = train_cfg.pop("optimizer", "Adam")
        scheduler = train_cfg.pop("scheduler", "StepLR")
        loss_fn = train_cfg.pop("loss_fn", "MSELoss")

        return GNSTraining(
            **train_cfg,
            optimizer=self._resolve_from_dict_or_torch(optimizer, _OPTIMIZERS, torch.optim),
            scheduler=self._resolve_from_dict_or_torch(scheduler, _SCHEDULERS, torch.optim.lr_scheduler),
            loss_fn=self._resolve_from_dict_or_torch(loss_fn, _LOSSES, torch.nn)(),
        )

    def resolve_optuna(self, optuna_dict: Dict) -> Dict:
        raw = optuna_dict.copy()
        study = raw.get("study")
        optimization_params = raw.get("optimization_params").copy()

        return {
            "optimization_params": optimization_params,
            "n_trials": study.get("n_trials", 100),
            "direction": study.get("direction", "minimize"),
            "seed": study.get("seed", 42),
            "pruner": _resolve_pruner(study.get("pruner")),
            "sampler": _resolve_sampler(study.get("sampler")),
            "save_dir": study.get("save_dir", None)
        }

    @property
    def model_params(self) -> dict:
        return asdict(self.model_cfg) if self.model_cfg else {}

    @property
    def training_params(self) -> dict:
        return asdict(self.training_cfg) if self.training_cfg else {}

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



def _resolve_sampler(sampler_cfg: Union[str, dict, None]):
    if sampler_cfg is None:
        return None
    if isinstance(sampler_cfg, str):
        return _SAMPLERS[sampler_cfg]()
    if isinstance(sampler_cfg, dict):
        sampler_type = sampler_cfg.pop("type", None)
        if sampler_type not in _SAMPLERS:
            raiseError(f"Unknown sampler type '{sampler_type}'")
        return _SAMPLERS[sampler_type](**sampler_cfg)
    raiseError("Invalid sampler configuration.")

def _resolve_pruner(pruner_cfg: Union[str, dict, None]):
    if pruner_cfg is None:
        return None
    if isinstance(pruner_cfg, str):
        return _PRUNERS[pruner_cfg]()
    if isinstance(pruner_cfg, dict):
        pruner_type = pruner_cfg.pop("type", None)
        if pruner_type not in _PRUNERS:
            raiseError(f"Unknown pruner type '{pruner_type}'")
        return _PRUNERS[pruner_type](**pruner_cfg)
    raiseError("Invalid pruner configuration.")

    
