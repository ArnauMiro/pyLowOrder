#!/usr/bin/env python

"""
Example: Training a GNS model on DLR airfoil data using pyLOM.

Updated to use the modern pyLOM API:
  - Dataclass-based GNS instantiation via GNSConfig
  - Optuna-compatible Pipeline execution
  - Hash-verified configuration saving

Author: Pablo Yeste
Date: 2025-08-01
"""

# ─────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────

from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from pyLOM.NN import Dataset, GNS, Pipeline, MinMaxScaler
from pyLOM.NN.optimizer import OptunaOptimizer
from pyLOM.NN.utils import RegressionEvaluator
from pyLOM.NN.utils.config_loader import load_full_config
from pyLOM.NN.utils.config_schema import GNSModelConfig, GNSTrainingConfig
from pyLOM.NN.utils.experiment import (
    save_experiment_artifacts,
    plot_training_and_validation_loss,
    plot_true_vs_pred,
    evaluate_dataset_with_metrics,
)
from pyLOM.utils import pprint
from pyLOM import cr_info

# ─────────────────────────────────────────────────────
# CONFIG LOAD, SEEDING, OUTPUT PATH
# ─────────────────────────────────────────────────────

config_path = Path("../pyLowOrder/Examples/NN/configs/gns_config.yaml").absolute()
config = load_full_config(
    path=config_path,
    model_registry={"gns": GNSModelConfig},
    training_registry={"default": GNSTrainingConfig},
)

# Typed config access
model_cfg = config.model.params
training_cfg = config.training
graph_path = config.model.graph_path
results_path = Path(config.experiment.results_path).absolute()
mode = config.experiment.mode
dataset_paths = config.datasets

# Flexible config for Optuna
optuna_study_cfg = config.optuna.study
optimization_params = config.optuna.optimization_params

# ─────────────────────────────────────────────────────
# DATASET LOADING AND SCALERS
# ─────────────────────────────────────────────────────

input_scaler = MinMaxScaler()
output_scaler = None  # Add output scaler here if required

ds_kwargs = dict(
    field_names=["CP"],
    add_variables=True,
    add_mesh_coordinates=False,
    variables_names=["AoA", "Mach"],
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    squeeze_last_dim=False,
)

ds_train = Dataset.load(dataset_paths["train_ds"], **ds_kwargs)
ds_val = Dataset.load(dataset_paths["val_ds"], **ds_kwargs)
ds_test = Dataset.load(dataset_paths["test_ds"], **ds_kwargs)

# ─────────────────────────────────────────────────────
# TRAINING OR OPTIMIZATION
# ─────────────────────────────────────────────────────

if mode == "optuna":
    optimizer = OptunaOptimizer(
        optimization_params=optimization_params,
        n_trials=optuna_study_cfg["n_trials"],
        direction=optuna_study_cfg.get("direction"),
        pruner=optuna_study_cfg.get("pruner"),
        sampler=optuna_study_cfg.get("sampler"),
        seed=optuna_study_cfg.get("seed"),
    )

    pipeline = Pipeline(
        train_dataset=ds_train,
        valid_dataset=ds_val,
        test_dataset=ds_test,
        optimizer=optimizer,
        model_class=GNS,
    )

else:
    model = GNS.from_graph_path(config=model_cfg, graph_path=graph_path)

    pipeline = Pipeline(
        train_dataset=ds_train,
        valid_dataset=ds_val,
        test_dataset=ds_test,
        model=model,
        training_cfg=training_cfg,
    )

logs = pipeline.run()

# ─────────────────────────────────────────────────────
# EVALUATION AND EXPERIMENT SAVING
# ─────────────────────────────────────────────────────

y_pred = pipeline.model.predict(ds_test).detach().cpu().numpy()
y_true = ds_test[:][1].detach().cpu().numpy()

pprint(0, f"\n>>> Predictions shape: {y_pred.shape}")
pprint(0, f">>> True values shape: {y_true.shape}")

evaluator = RegressionEvaluator()
metrics = evaluator(y_true, y_pred)
evaluator.print_metrics()
logs["metrics"] = metrics

save_path = save_experiment_artifacts(
    base_path=results_path,
    model=pipeline.model,
    metrics_dict=metrics,
    input_scaler=input_scaler,
    output_scaler=output_scaler,
    extra_files={
        "train_test_loss.png": lambda p: plot_training_and_validation_loss(
            logs["train_loss"], logs["test_loss"], p
        ),
        "true_vs_pred.png": lambda p: plot_true_vs_pred(y_true, y_pred, p),
    },
    return_path=True,
)

pprint(0, f"\nExperiment artifacts saved to: {save_path}")

# ─────────────────────────────────────────────────────
# INFERENCE TEST
# ─────────────────────────────────────────────────────

# Reload model and input scaler
model_reloaded = GNS.from_graph_path(config=model_cfg, graph_path=graph_path)
input_scaler_reloaded = MinMaxScaler.from_dict(input_scaler.to_dict())
model_reloaded.eval()

# Manual input
sample_input = np.array([[4.0, 0.7]])  # AoA, Mach
scaled_input = input_scaler_reloaded.transform(sample_input)
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
# DEBUGGING: RELOAD FROM DISK
# ─────────────────────────────────────────────────────

save_path = Path("/home/p.yeste/CETACEO_RESULTS/nlr7301/2025-08-04T05-24-09_GNS/")

print("\n>>> Reloading model and input scaler from disk...")
model_reloaded = GNS.load(save_path / "model.pth")
with open(save_path / "input_scaler.pkl", "rb") as f:
    input_scaler_reloaded = pickle.load(f)

# Internal evaluation on training set
loss_internal = model_reloaded._run_epoch(
    input_dataloader=model_reloaded._helpers.init_dataloader(ds_train, batch_size=1),
    subgraph_loader=model_reloaded._helpers.init_subgraph_dataloader(batch_size=256),
    loss_fn=torch.nn.MSELoss(),
    return_loss=True,
    is_train=False,
)
print(f"[Internal Eval on train] MSE: {loss_internal:.6f}")

# External evaluation on training set
preds = model_reloaded.predict(ds_train).detach().cpu().numpy()
_, targets = ds_train[:]
if torch.is_tensor(targets):
    targets = targets.cpu().numpy()

metrics = evaluate_dataset_with_metrics(preds, targets)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ─────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────

cr_info()
