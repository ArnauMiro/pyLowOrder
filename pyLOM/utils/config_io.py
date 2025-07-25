#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Utils - Configuration I/O utilities.
#
# Last rev: 25/07/2025

import yaml
from pathlib import Path
from typing import Union

from pyLOM.NN.utils.configs import GNSConfig

def load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML file into a Python dictionary."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def build_GNS_config(config: dict) -> GNSConfig:
    model_dict = config["model"].copy()
    model_dict["device"] = config["experiment"].get("device", "cuda")
    model_dict["seed"] = config["experiment"].get("seed", None)
    return GNSConfig(**model_dict)