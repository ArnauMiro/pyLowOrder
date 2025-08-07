import yaml
import importlib
from dacite import from_dict, Config
from typing import Any, Callable, Type
import torch


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
    from config_schema import AppConfig, ModelSection  # defined below

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    type_hooks = GeneralTypeHooks()
    dacite_cfg = Config(type_hooks=type_hooks, strict=True)

    # Dispatch model.params to proper subclass
    model_type = raw["model"].get("type")
    if model_type is None:
        raise ValueError("Missing 'model.type' in YAML.")

    model_cls = model_registry[model_type.lower()]
    model_params = from_dict(data_class=model_cls, data=raw["model"]["params"], config=dacite_cfg)
    raw["model"]["params"] = model_params

    # Dispatch training to proper subclass
    training_type = raw.get("training_type", "default")
    training_cls = training_registry[training_type.lower()]
    raw["training"] = from_dict(data_class=training_cls, data=raw["training"], config=dacite_cfg)

    return from_dict(data_class=AppConfig, data=raw, config=dacite_cfg)
