from dataclasses import is_dataclass, fields
from functools import wraps
from typing import Type, Callable

def accepts_config(config_class: Type) -> Callable:
    """
    Decorator that builds or validates a config object for a method.

    Supports both:
      - Passing config=GNSFitConfig(...) directly
      - Passing **kwargs to be converted into GNSFitConfig

    Args:
        config_class (Type): A dataclass type to instantiate.

    Raises:
        TypeError: If provided config is not an instance of config_class.
        ValueError: If invalid kwargs are provided.

    Returns:
        Callable: Wrapped function with guaranteed `config` instance.
    """
    if not is_dataclass(config_class):
        raise TypeError(f"{config_class} is not a dataclass.")

    config_fields = {f.name for f in fields(config_class)}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Case A: config is provided directly
            if "config" in kwargs:
                config = kwargs.pop("config")
                if not isinstance(config, config_class):
                    raise TypeError(f"Expected `config` to be instance of {config_class.__name__}, got {type(config).__name__}")
            else:
                # Case B: construct from kwargs
                config_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in config_fields}
                try:
                    config = config_class(**config_kwargs)
                except TypeError as e:
                    raise ValueError(f"Invalid arguments for {config_class.__name__}: {e}")

            return func(*args, config=config, **kwargs)

        # Augment docstring
        doc = func.__doc__ or ""
        doc += f"\n\nAccepted config fields for `{config_class.__name__}`:\n"
        for f in fields(config_class):
            doc += f"    - {f.name} ({f.type.__name__ if hasattr(f.type, '__name__') else f.type})\n"
        wrapper.__doc__ = doc

        return wrapper

    return decorator
