#!/usr/bin/env python
"""
Example: Training a GNS model on DLR/NLR airfoil data using pyLOM.

Modernized for DTO-based configs:
  - Dataclass-based GNS instantiation via GNSModelConfig / GNSTrainingConfig
  - Optuna workflow using pure dict search spaces
  - Provenance via graph_path (outside the model DTO)
  - Fingerprint checked internally by GNS

Author: Pablo Yeste
Date: 2025-08-01
"""

# ─────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────
from pathlib import Path
import numpy as np
import torch

from dacite import from_dict, Config as DaciteConfig

from pyLOM.NN import Dataset, GNS, Pipeline, MinMaxScaler
from pyLOM.NN.optimizer import OptunaOptimizer
from pyLOM.NN.utils import RegressionEvaluator
from pyLOM.NN.utils.config_schema import (
    GNSModelConfig,
    GNSTrainingConfig,
)
from pyLOM.NN.utils.experiment import (
    save_experiment_artifacts,
    plot_training_and_validation_loss,
    plot_true_vs_pred,
)
from pyLOM.utils.config_resolvers import load_yaml, instantiate_from_config
from pyLOM.utils import pprint, raiseError
from pyLOM import cr_info

# ─────────────────────────────────────────────────────
# CONFIG LOAD, DTOs, OUTPUT PATH
# ─────────────────────────────────────────────────────

config_path = Path("../pyLowOrder/Examples/NN/configs/gns_config.yaml").absolute()
cfg = load_yaml(config_path)

# Experiment section
exp_cfg = cfg["experiment"]
results_path = Path(exp_cfg["results_path"]).absolute()
mode = exp_cfg.get("mode", "train")

# Datasets section
dataset_paths = cfg["datasets"]

# Model DTO + provenance (graph_path outside the DTO)
graph_path = cfg["model"]["graph_path"]
model_cfg_dict = cfg["model"]["config"]
dacite_cfg = DaciteConfig(strict=True)
model_cfg = from_dict(GNSModelConfig, model_cfg_dict, config=dacite_cfg)

# Training DTO
training_cfg_dict = cfg["training"]
training_cfg = from_dict(GNSTrainingConfig, training_cfg_dict, config=dacite_cfg)

# Optional Optuna section (search space is not DTO; it’s a free-form dict)
optuna_cfg = cfg.get("optuna", {})
optuna_study_cfg = optuna_cfg.get("study", {}) or {}
optimization_params = optuna_cfg.get("optimization_params", {}) or {}

# ─────────────────────────────────────────────────────
# DATASET LOADING AND SCALERS
# ─────────────────────────────────────────────────────

inputs_scaler = MinMaxScaler()
outputs_scaler = None  # add an output scaler if required

ds_kwargs = dict(
    field_names=["CP"],
    add_variables=True,
    add_mesh_coordinates=False,
    variables_names=["AoA", "Mach"],
    inputs_scaler=inputs_scaler,
    outputs_scaler=outputs_scaler,
    squeeze_last_dim=False,
)

ds_train = Dataset.load(dataset_paths["train_ds"], **ds_kwargs)
ds_val   = Dataset.load(dataset_paths["val_ds"],   **ds_kwargs)
ds_test  = Dataset.load(dataset_paths["test_ds"],  **ds_kwargs)

# ─────────────────────────────────────────────────────
# TRAINING OR OPTIMIZATION
# ─────────────────────────────────────────────────────

if mode == "optuna":
    # Instantiate Optuna components from strings in YAML (if present)
    pruner  = instantiate_from_config(optuna_study_cfg.get("pruner"))
    sampler = instantiate_from_config(optuna_study_cfg.get("sampler"))

    optimizer = OptunaOptimizer(
        optimization_params=optimization_params["optimization_params"] if "optimization_params" in optimization_params else optimization_params,  # tolerate nesting
        n_trials=optuna_study_cfg.get("n_trials", 50),
        direction=optuna_study_cfg.get("direction", "minimize"),
        pruner=pruner,
        sampler=sampler,
        seed=optuna_study_cfg.get("seed", 42),
        save_dir=str(results_path),
    )

    # Pipeline expected to drive GNS.create_optimized_model under the hood
    pipeline = Pipeline(
        train_dataset=ds_train,
        valid_dataset=ds_val,
        test_dataset=ds_test,
        optimizer=optimizer,
        model_class=GNS,
    )

elif mode == "train":
    model = GNS.from_graph_path(config=model_cfg, graph_path=graph_path)

    pipeline = Pipeline(
        train_dataset=ds_train,
        valid_dataset=ds_val,
        test_dataset=ds_test,
        model=model,
        training_params={"config": training_cfg},  # <- clave correcta
    )

else:
    raiseError(f"Unknown mode: {mode}. Use 'train' or 'optuna'.")

logs = pipeline.run()

# ─────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────

y_pred = pipeline.model.predict(ds_test).detach().cpu().numpy()
y_true = ds_test[:][1].detach().cpu().numpy()

pprint(0, f"\n>>> Predictions shape: {y_pred.shape}")
pprint(0, f">>> True values shape: {y_true.shape}")

evaluator = RegressionEvaluator()
metrics = evaluator(y_true, y_pred)
evaluator.print_metrics()
logs["metrics"] = metrics

# ─────────────────────────────────────────────────────
# EXPERIMENT SAVING
# ─────────────────────────────────────────────────────
save_path = save_experiment_artifacts(
    base_path=results_path,
    model=pipeline.model,
    metrics_dict=metrics,
    inputs_scaler=inputs_scaler,
    outputs_scaler=outputs_scaler,
    extra_files={
        "train_val_loss.png": lambda p: plot_training_and_validation_loss(
            logs.get("train_loss", []), logs.get("test_loss", []), p
        ),
        "true_vs_pred.png": lambda p: plot_true_vs_pred(y_true, y_pred, p),
    },
    return_path=True,
)

# ─────────────────────────────────────────────────────
# INFERENCE TEST
# ─────────────────────────────────────────────────────

# Reload model and input scaler
model_reloaded = GNS.from_graph_path(config=model_cfg, graph_path=graph_path)
inputs_scaler_reloaded = MinMaxScaler.load(str(save_path / "inputs_scaler.json"))

model_reloaded.eval()

# Manual input
sample_input = np.array([[4.0, 0.7]], dtype=np.float32)  # AoA, Mach
scaled_input = inputs_scaler_reloaded.transform(sample_input)
input_tensor = torch.tensor(scaled_input, dtype=torch.float32, device=model_reloaded.device)

with torch.no_grad():
    prediction = model_reloaded.predict(input_tensor)

pprint(0, f">>> Prediction shape: {prediction.shape}")
pprint(0, f">>> First 5 values: {prediction[0, :5].detach().cpu().numpy()}")

# Compare with nearest test sample
X_test, Y_test = ds_test[:]
nearest_idx = np.argmin(np.linalg.norm(np.array(X_test) - np.array(sample_input), axis=1))
reference = Y_test[nearest_idx]

pprint(0, f">>> Comparing with test sample at index {nearest_idx}")
plot_true_vs_pred(reference, prediction[0].cpu().numpy())

# ─────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────

cr_info()
