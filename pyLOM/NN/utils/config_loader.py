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
from ...utils import raiseError


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
def auto_instantiate(obj: Any) -> Any:
    """
    Recursively instantiate objects from dicts containing a `type` key.
    - `args` is a list of positional args.
    - Any other key is treated as a kwarg.
    """
    if isinstance(obj, dict):
        if "type" in obj:
            cls = resolve_callable(obj["type"]) if isinstance(obj["type"], str) else obj["type"]

            args = [auto_instantiate(a) for a in obj.get("args", [])]
            kwargs = {k: auto_instantiate(v) for k, v in obj.items() if k not in ("type", "args")}

            return cls(*args, **kwargs)
        else:
            return {k: auto_instantiate(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [auto_instantiate(v) for v in obj]

    elif isinstance(obj, str):
        if "." in obj:
            try:
                return resolve_callable(obj)
            except Exception:
                return obj
        return obj

    return obj



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
def load_full_config(path: Union[str, Path]) -> AppConfig:
    """Load YAML and instantiate all sections using `auto_instantiate`."""
    raw = load_yaml(path)
    resolved = auto_instantiate(raw)
    type_hooks = GeneralTypeHooks()
    return from_dict(data_class=AppConfig, data=resolved, config=Config(type_hooks=type_hooks, strict=True))

