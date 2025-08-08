import yaml
import importlib
from pathlib import Path
from typing import Any, Type, Union, get_origin, get_args
from dataclasses import asdict, is_dataclass
from inspect import isclass
import types
import torch
import torch.nn as nn
from dacite import from_dict, Config

from .config_schema import AppConfig


# ==========================================================
# Core resolution utilities
# ==========================================================
def get_path(obj: Any) -> str:
    """Return full import path for a class or instance."""
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__name__}"
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def resolve_callable(value: str) -> Any:
    """Resolve a dotted import path string to a Python object."""
    module_path, attr = value.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


# ==========================================================
# Type hooks for dacite
# ==========================================================
class GeneralTypeHooks(dict):
    """
    Dynamic type_hooks mapping for dacite.
    Handles:
    - type[...] and Type[...] annotations
    - Optional/Union containing type[...] arms
    - torch.device
    """

    def __missing__(self, key):
        origin = get_origin(key)

        # Plain `type`
        if key is type:
            hook = lambda v: resolve_callable(v) if isinstance(v, str) else v
            self[key] = hook
            return hook

        # Generic type[...] / Type[...]
        if origin is type:
            hook = lambda v: resolve_callable(v) if isinstance(v, str) else v
            self[key] = hook
            return hook

        # Optional/Union containing a type[...] arm
        if origin in (Union, getattr(types, "UnionType", None)):
            if any(a is type or get_origin(a) is type for a in get_args(key)):
                hook = lambda v: resolve_callable(v) if isinstance(v, str) else v
                self[key] = hook
                return hook

        # torch.device
        if key is torch.device:
            hook = lambda v: torch.device(v)
            self[key] = hook
            return hook

        # Default passthrough
        self[key] = lambda v: v
        return self[key]


# ==========================================================
# Bidirectional config transformation
# ==========================================================
def resolve_config_dict(cfg: dict) -> dict:
    """
    Recursively resolve string references in a config dict:
    - "torch.nn.ReLU" → class
    - "cuda"/"cpu" → torch.device
    - filesystem paths → Path objects
    """
    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            resolved[key] = resolve_config_dict(value)
        elif isinstance(value, str):
            if key == "device" and value in ("cuda", "cpu"):
                resolved[key] = torch.device(value)
            elif "." in value:  # Try to import
                try:
                    obj = resolve_callable(value)
                    # NEW: If it's a nn.Module subclass, instantiate it
                    if isclass(obj) and issubclass(obj, torch.nn.Module):
                        resolved[key] = obj()
                    else:
                        resolved[key] = obj
                except Exception:
                    resolved[key] = value
            elif "/" in value or value.endswith(".h5"):
                resolved[key] = Path(value)
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved


def serialize_config_dict(cfg: dict) -> dict:
    """
    Recursively serialize config by converting objects/classes
    into full import path strings.
    """
    serialized = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            serialized[key] = serialize_config_dict(value)
        elif is_dataclass(value):
            serialized[key] = serialize_config_dict(asdict(value))
        else:
            try:
                serialized[key] = get_path(value)
            except Exception:
                serialized[key] = value
    return serialized


# ==========================================================
# YAML utilities
# ==========================================================
def load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML file into a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ==========================================================
# Dataclass helpers
# ==========================================================
def to_dataclass(data: dict, cls: Type) -> Any:
    """Instantiate a dataclass from a resolved config dict."""
    type_hooks = GeneralTypeHooks()
    return from_dict(data_class=cls, data=data, config=Config(type_hooks=type_hooks, strict=True))


def from_dataclass(instance: Any) -> dict:
    """Serialize a dataclass into a plain dict with import paths."""
    if not is_dataclass(instance):
        raise ValueError("from_dataclass expects a dataclass instance")
    return serialize_config_dict(asdict(instance))


# ==========================================================
# Full config loader (App-level)
# ==========================================================
def load_full_config(
    path: Union[str, Path],
    model_registry: dict[str, Type],
    training_registry: dict[str, Type]
) -> Any:
    """
    Load full application config from YAML and instantiate
    the corresponding AppConfig dataclass.
    """
    raw = load_yaml(path)

    # Pre-resolve strings into proper Python objects
    raw["model"]["params"] = resolve_config_dict(raw["model"]["params"])
    raw["training"] = resolve_config_dict(raw["training"])

    dacite_cfg = Config(type_hooks={torch.device: torch.device}, strict=True)

    # Build model config
    model_cls = model_registry[raw["model"]["type"].lower()]
    raw["model"]["params"] = from_dict(data_class=model_cls, data=raw["model"]["params"], config=dacite_cfg)

    # Build training config
    training_cls = training_registry[raw.get("training_type", "default").lower()]
    raw["training"] = from_dict(data_class=training_cls, data=raw["training"], config=dacite_cfg)

    # Build final AppConfig
    return from_dict(data_class=AppConfig, data=raw, config=dacite_cfg)
