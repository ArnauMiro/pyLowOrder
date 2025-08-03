from dataclasses import is_dataclass, fields
from functools import wraps
from typing import Type, Callable, Any


def accepts_config(config_class: Type) -> Callable:
    """
    Decorator that ensures a function receives a `config` instance of `config_class`.

    Accepts either:
      - `config=instance_of_config_class`
      - `config=dict(...)` with fields matching the config class
      - `**kwargs` that include config-class fields

    Args:
        config_class (Type): A dataclass type to enforce or construct.

    Returns:
        Callable: Wrapped function with guaranteed `config: config_class` in kwargs.
    """
    if not is_dataclass(config_class):
        raise TypeError(f"{config_class} must be a dataclass.")

    config_fields = {f.name for f in fields(config_class)}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Case A: config provided explicitly
            if "config" in kwargs:
                raw = kwargs.pop("config")

                if isinstance(raw, config_class):
                    config = raw
                elif isinstance(raw, dict):
                    try:
                        config = config_class(**raw)
                    except TypeError as e:
                        raise ValueError(f"Invalid config dictionary for {config_class.__name__}: {e}")
                else:
                    raise TypeError(f"`config` must be {config_class.__name__} or dict, got {type(raw).__name__}")

            else:
                # Case B: construct from kwargs
                config_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in config_fields}
                try:
                    config = config_class(**config_kwargs)
                except TypeError as e:
                    raise ValueError(f"Invalid arguments for {config_class.__name__}: {e}")

            return func(*args, config=config, **kwargs)

        # Augment docstring with config fields
        doc = func.__doc__ or ""
        doc += f"\n\nAccepted config fields for `{config_class.__name__}`:\n"
        for f in fields(config_class):
            doc += f"    - {f.name} ({getattr(f.type, '__name__', f.type)})\n"
        wrapper.__doc__ = doc

        return wrapper

    return decorator
