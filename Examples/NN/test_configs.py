#!/usr/bin/env python
"""
Script to manually inspect a GNSConfig instance loaded from YAML.

This is intended for debugging and interactive inspection without using pytest.
"""
#%%
from pathlib import Path
from pprint import pprint
from dataclasses import asdict, is_dataclass
import dacite

from pyLOM.NN.utils.config_manager import load_yaml, resolve_config_dict, GNSConfig
from pyLOM.NN.gns import GNSModelConfig, GNSTrainingConfig

#%%
# Load config from YAML
config_path = Path("./configs/gns_config.yaml").absolute()
cfg_serial = load_yaml(config_path)
cfg = resolve_config_dict(cfg_serial)

graph_path = cfg["model"]["graph_path"]
model_cfg = dacite.from_dict(GNSModelConfig, cfg["model"]["params"])
training_cfg = dacite.from_dict(GNSTrainingConfig, cfg["training"])
optuna_cfg = cfg.get("optuna", {})


pprint("🔧 Loaded GNSConfig from YAML:")
# Check dataclass status
print(f"  model is dataclass: {is_dataclass(model_cfg)}")
print(f"  training is dataclass: {is_dataclass(training_cfg)}")
print(f"  optuna is dataclass: {is_dataclass(optuna_cfg)}")
print(f"  graph_path: {graph_path}")
print()

pprint(cfg_serial)







#%%
print("✅ Successfully loaded GNSConfig from YAML.")
print()

# Inspect top-level structure
print("🔍 Top-level sections:")
print(f"  model: {type(cfg.model).__name__}")
print(f"  training: {type(cfg.training).__name__}")
print(f"  optuna: {type(cfg.optuna).__name__ if cfg.optuna else 'None'}")
print()

# Check dataclass status
print("🧪 Dataclass check:")
print(f"  model is dataclass: {is_dataclass(cfg.model)}")
print(f"  training is dataclass: {is_dataclass(cfg.training)}")
print(f"  training['dataloader'] is dataclass: {is_dataclass(cfg.training.dataloader)}")
print(f"  training['subgraph_loader'] is dataclass: {is_dataclass(cfg.training.subgraph_loader)}")

# Inspect model section
print("📦 Model config:")
pprint(asdict(cfg.model))
print()

# Inspect training section
print("🎯 Training config:")
pprint(asdict(cfg.training))
print()

# Inspect optuna (if any)
print("📊 Optuna config:")
pprint(cfg.optuna)
print()
