# utils/config_resolver.py

import importlib
import torch
from torch import nn, optim
from typing import Any, Mapping, Optional
from ...utils import raiseError

def resolve_import(path: str):
    """Import a symbol from a fully-qualified import path."""
    try:
        module, name = path.rsplit(".", 1)
        return getattr(importlib.import_module(module), name)
    except Exception:
        raiseError(f"Cannot import '{path}'.")

def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    return torch.device(device_str)

def resolve_activation(activation_path: str) -> nn.Module:
    """Instantiate an activation nn.Module from an import path."""
    cls = resolve_import(activation_path)
    obj = cls()
    if not isinstance(obj, nn.Module):
        raiseError(f"Activation '{activation_path}' is not an nn.Module.")
    return obj

def resolve_loss(loss_path: str) -> nn.Module:
    """Instantiate a loss nn.Module from an import path."""
    cls = resolve_import(loss_path)
    obj = cls()
    if not isinstance(obj, nn.Module):
        raiseError(f"Loss '{loss_path}' is not an nn.Module.")
    return obj

def resolve_optimizer(optimizer_path: str):
    """Resolve optimizer class from an import path."""
    cls = resolve_import(optimizer_path)
    if not isinstance(cls, type) or not issubclass(cls, optim.Optimizer):
        raiseError(f"Optimizer '{optimizer_path}' is not a torch.optim.Optimizer.")
    return cls

def resolve_scheduler(scheduler_path: str):
    """Resolve scheduler class from an import path."""
    if scheduler_path is None:
        return None
    cls = resolve_import(scheduler_path)
    # _LRScheduler is internal; comprobamos interfaz básica
    if not isinstance(cls, type):
        raiseError(f"Scheduler '{scheduler_path}' must be a class.")
    return cls

# pyLOM/utils/factory.py


def instantiate_from_config(spec: Optional[Mapping[str, Any]]) -> Any:
    """
    Instantiate a Python object from a config dict with a `type` key.

    Example spec:
    --------
    spec = {
        "type": "torch.optim.Adam",
        "args": [[...]],          # optional positional args (list)
        "lr": 0.001,              # keyword args
        "weight_decay": 1e-4
    }

    Returns
    -------
    object : Any
        Instantiated object.
    """
    if not spec:
        return None
    if "type" not in spec:
        raise ValueError("Spec must contain a 'type' key.")

    cls_or_fn = resolve_import(spec["type"])

    args = spec.get("args", [])
    if not isinstance(args, (list, tuple)):
        raise TypeError("'args' must be a list or tuple if provided.")

    kwargs = {k: v for k, v in spec.items() if k not in ("type", "args")}
    return cls_or_fn(*args, **kwargs)

