from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path
import yaml
import dacite
from importlib import import_module
from inspect import isclass, isdataclass
from dataclasses import asdict

from ..gns import GNSModelConfig, GNSTrainingConfig, SubgraphDataloaderConfig

#-----------------------------------------------------------------------
# Utility functions for resolving and serializing configuration objects
#-----------------------------------------------------------------------

def resolve_object(path: str) -> object:
    """
    Dynamically import an object from a dotted string path.

    Args:
        path (str): Full dotted path to the object, e.g., 'torch.optim.Adam'.

    Returns:
        object: The resolved Python object (class, function, etc.).
    """
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid path '{path}'. Must be a full import path.")
    module = import_module(module_path)
    return getattr(module, attr)


def resolve_config_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively resolve all string values that represent import paths (e.g., 'torch.nn.ELU')
    into the corresponding Python objects.
    """
    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            resolved[key] = resolve_config_dict(value)
        elif isinstance(value, str) and "." in value:
            try:
                resolved[key] = resolve_object(value)
            except Exception:
                resolved[key] = value  # Keep string if it doesn't resolve (e.g. enums, names)
        else:
            resolved[key] = value
    return resolved


def serialize_config_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively serialize a config dict by converting objects or classes
    into their full import paths as strings.
    """
    def get_path(obj: Any) -> str:
        if isclass(obj) or callable(obj):
            return f"{obj.__module__}.{obj.__name__}"
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

    serialized = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            serialized[key] = serialize_config_dict(value)
        elif isdataclass(value):
            serialized[key] = serialize_config_dict(asdict(value))
        else:
            try:
                serialized[key] = get_path(value)
            except Exception:
                serialized[key] = value
    return serialized


#----------------------------------------------------------------------
# Base class for all model configurations
#----------------------------------------------------------------------


class ModelConfigBase(ABC):
    """
    Base class for all model configuration wrappers.

    Subclasses must define:
        - model: a dataclass with model hyperparameters
        - training: a dataclass with training parameters
        - optuna: optional search space (raw dict)
    """

    @abstractmethod
    def __init__(self, model: Any, training: Any, optuna: Optional[Dict[str, Any]] = None):
        self.model = model
        self.training = training
        self.optuna = optuna or {}

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfigBase":
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls.from_dict(raw_cfg)

    @classmethod
    def from_dict(cls, raw_cfg: Dict[str, Any]) -> "ModelConfigBase":
        raise NotImplementedError("Subclasses must implement from_dict()")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": {
                "params": serialize_config_dict(self.model)
            },
            "training": serialize_config_dict(self.training),
            "optuna": self.optuna
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


#----------------------------------------------------------------------
# Model-specific configuration class
#----------------------------------------------------------------------

class GNSConfig(ModelConfigBase):
    def __init__(self, model: GNSModelConfig, training: GNSTrainingConfig, optuna: Optional[Dict[str, Any]] = None):
        super().__init__(model=model, training=training, optuna=optuna)

    @classmethod
    def from_dict(cls, raw_cfg: Dict[str, Any]) -> "GNSConfig":
        resolved = resolve_config_dict(raw_cfg)

        model_cfg = resolved["model"]["params"]
        training_cfg = resolved["training"]
        optuna_cfg = raw_cfg.get("optuna", {})

        model = dacite.from_dict(GNSModelConfig, model_cfg)
        training = dacite.from_dict(GNSTrainingConfig, training_cfg)

        return cls(model=model, training=training, optuna=optuna_cfg)

