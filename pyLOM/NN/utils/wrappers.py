from functools import wraps
from dataclasses import fields, is_dataclass
from typing import Callable, Type


def accepts_config(config_class: Type) -> Callable:
    """
    Decorator that builds a config object from **kwargs using a dataclass.

    Args:
        config_class (Type): A dataclass to instantiate (e.g., GNSTrainingConfig)

    Raises:
        ValueError: If invalid kwargs are provided

    Returns:
        Callable: Wrapped method with a `config` argument injected
    """
    if not is_dataclass(config_class):
        raise TypeError(f"{config_class} is not a dataclass.")

    config_field_names = {f.name for f in fields(config_class)}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in config_field_names}
            try:
                config = config_class(**config_kwargs)
            except TypeError as e:
                raise ValueError(f"Invalid arguments for {config_class.__name__}: {e}")
            return func(*args, config=config, **kwargs)

        doc = func.__doc__ or ""
        doc += f"\n\nAccepted `**kwargs` for `{config_class.__name__}`:\n"
        for f in fields(config_class):
            doc += f"    - {f.name} ({f.type.__name__ if hasattr(f.type, '__name__') else f.type})\n"
        wrapper.__doc__ = doc

        return wrapper

    return decorator
