import yaml
import importlib
from pathlib import Path
from typing import Any, Callable, Type, Union
from dataclasses import asdict, is_dataclass
from inspect import isclass
import torch
from dacite import from_dict, Config


# ------------------------------
# Core resolution functions
# ------------------------------
def resolve_callable(value: str) -> Any:
    """Resolve a string like 'torch.nn.ReLU' to the actual class or function."""
    module_path, attr = value.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def resolve_if_string(value: Any, expected_type: Type) -> Any:
    """Generic resolver for strings to classes/functions/devices."""
    if isinstance(value, str):
        if expected_type == torch.device:
            return torch.device(value)
        if expected_type in [type, Callable] or isinstance(expected_type, type):
            return resolve_callable(value)
    return value


class GeneralTypeHooks(dict):
    """Custom type_hooks dict for dacite that creates hooks dynamically."""
    def __missing__(self, key):
        def hook(value):
            return resolve_if_string(value, key)
        self[key] = hook
        return hook


# ------------------------------
# Bidirectional conversion
# ------------------------------
def resolve_config_dict(cfg: dict) -> dict:
    """
    Recursively resolve string references in the config dict:
    - "torch.nn.ReLU" -> class
    - "cuda"/"cpu" -> torch.device
    - paths -> Path objects
    """
    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            resolved[key] = resolve_config_dict(value)
        elif isinstance(value, str):
            if key == "device" and value in ("cuda", "cpu"):
                resolved[key] = torch.device(value)
            elif "." in value:
                try:
                    resolved[key] = resolve_callable(value)
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
    Recursively serialize a config dict by converting objects/classes
    into full import path strings.
    """
    def get_path(obj: Any) -> str:
        if isclass(obj) or callable(obj):
            return f"{obj.__module__}.{obj.__name__}"
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

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


# ------------------------------
# YAML utilities
# ------------------------------
def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------
# Dataclass instantiation helpers
# ------------------------------
def to_dataclass(data: dict, cls: Type) -> Any:
    """Build dataclass from raw (resolved) dict."""
    type_hooks = GeneralTypeHooks()
    return from_dict(data_class=cls, data=data, config=Config(type_hooks=type_hooks, strict=True))


def from_dataclass(instance: Any) -> dict:
    """Serialize a dataclass into a clean dict with import-paths."""
    if not is_dataclass(instance):
        raise ValueError("from_dataclass expects a dataclass instance")
    return serialize_config_dict(asdict(instance))


# ------------------------------
# Full config loader (App-level)
# ------------------------------
def load_full_config(path: str, model_registry: dict[str, Type], training_registry: dict[str, Type]) -> Any:
    """
    Load full application config from YAML, dispatching to correct model/training subclasses.

    Args:
        path: Path to the YAML file.
        model_registry: Mapping from model type names to ModelConfig subclasses.
        training_registry: Mapping from training type names to TrainingConfig subclasses.

    Returns:
        An instance of AppConfig with all fields fully resolved.
    """
    from config_schema import AppConfig  # Assumes available elsewhere

    raw = load_yaml(path)

    type_hooks = GeneralTypeHooks()
    dacite_cfg = Config(type_hooks=type_hooks, strict=True)

    # Dispatch model.params
    model_type = raw["model"].get("type")
    if model_type is None:
        raise ValueError("Missing 'model.type' in YAML config")
    model_cls = model_registry[model_type.lower()]
    raw["model"]["params"] = from_dict(model_cls, raw["model"]["params"], dacite_cfg)

    # Dispatch training
    training_type = raw.get("training_type", "default")
    training_cls = training_registry[training_type.lower()]
    raw["training"] = from_dict(training_cls, raw["training"], dacite_cfg)

    return from_dict(AppConfig, raw, dacite_cfg)
