from dataclasses import is_dataclass, fields
from functools import wraps
from typing import Type, Callable, Any


def config_from_kwargs(config_class: Type) -> Callable:
    """
    Decorator that allows a function to accept a configuration either as:
      - `config=instance_of_config_class`
      - `config=dict(...)` (converted to instance)
      - `**kwargs` matching the dataclass fields

    Guarantees that the function receives a `config: config_class` in kwargs.
    """
    if not is_dataclass(config_class):
        raise TypeError(f"{config_class} must be a dataclass.")

    config_fields = {f.name for f in fields(config_class)}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if "config" in kwargs:
                if any(k in config_fields for k in kwargs):
                    raise ValueError("Cannot mix `config=...` with individual config fields as kwargs.")
                raw = kwargs.pop("config")
                if isinstance(raw, config_class):
                    config = raw
                elif isinstance(raw, dict):
                    config = config_class(**raw)
                else:
                    raise TypeError(f"`config` must be {config_class.__name__} or dict.")
            else:
                config_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in config_fields}
                config = config_class(**config_kwargs)

            return func(*args, config=config, **kwargs)

        # Docstring augmentation
        doc = func.__doc__ or ""
        doc += f"\n\nAccepted config fields for `{config_class.__name__}`:\n"
        for f in fields(config_class):
            doc += f"    - {f.name} ({getattr(f.type, '__name__', f.type)})\n"
        wrapper.__doc__ = doc

        return wrapper

    return decorator

