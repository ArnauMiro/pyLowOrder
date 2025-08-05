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
import torch
import pickle

import torch.nn.functional as F

from pyLOM.NN import Dataset, GNS, Pipeline, MinMaxScaler
from pyLOM.NN.optimizer import OptunaOptimizer
from pyLOM.NN.utils import RegressionEvaluator
from pyLOM.NN.utils.config_manager import load_yaml, deserialize_config
from pyLOM import cr_info
from pyLOM.NN.utils.experiment import save_experiment_artifacts, plot_training_and_validation_loss, plot_true_vs_pred, evaluate_dataset_with_metrics

# ─────────────────────────────────────────────────────
# SETUP: CONFIG LOAD, SEEDING, OUTPUT PATH
# ─────────────────────────────────────────────────────

config_path = Path("../pyLowOrder/Examples/NN/configs/gns_config.yaml").absolute()
raw_cfg = load_yaml(config_path)

results_dir = Path(raw_cfg["experiment"]["results_dir"]).absolute()
results_dir.mkdir(parents=True, exist_ok=True)

dataset_paths = raw_cfg["datasets"]
mode = raw_cfg["experiment"].get("mode", "train")

model_cfg_raw = raw_cfg["model"]
training_cfg_raw = raw_cfg["training"]
optuna_cfg_raw = raw_cfg["optuna"]

model_params_cfg = model_cfg_raw["params"]
graph_path = model_cfg_raw["graph_path"]

model_params = deserialize_config(model_params_cfg)
training_params = deserialize_config(training_cfg_raw)
optuna_params = deserialize_config(optuna_cfg_raw)

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
    squeeze_last_dim=False
)

ds_train = Dataset.load(dataset_paths["train_ds"], **ds_kwargs)
ds_val   = Dataset.load(dataset_paths["val_ds"], **ds_kwargs)
ds_test  = Dataset.load(dataset_paths["test_ds"], **ds_kwargs)


# ─────────────────────────────────────────────────────
# TRAINING OR OPTIMIZATION
# ─────────────────────────────────────────────────────

if mode == "optuna":
    study = optuna_params["study"]
    optimizer = OptunaOptimizer(
        optimization_params=optuna_params["optimization_params"],
        n_trials=study["n_trials"],
        direction=study.get("direction"),
        pruner=study.get("pruner"),
        sampler=study.get("sampler"),
        seed=study.get("seed"),
    )

    pipeline = Pipeline(
        train_dataset=ds_train,
        valid_dataset=ds_val,
        test_dataset=ds_test,
        optimizer=optimizer,
        model_class=GNS,
    )

else:
    model = GNS.from_graph_path(config=model_params, graph_path=graph_path)

    pipeline = Pipeline(
        train_dataset=ds_train,
        valid_dataset=ds_val,
        test_dataset=ds_test,
        model=model,
        training_params=training_params,
    )

# logs = pipeline.run()


# ─────────────────────────────────────────────────────
# EVALUATION AND EXPERIMENT SAVING
# ─────────────────────────────────────────────────────

# y_pred = pipeline.model.predict(ds_test)
# y_pred = y_pred.detach().cpu().numpy()
# y_true = ds_test[:][1]
# y_true = y_true.detach().cpu().numpy()

# print(f"\n>>> Predictions shape: {y_pred.shape}")
# print(f">>> True values shape: {y_true.shape}")


# evaluator = RegressionEvaluator()
# metrics = evaluator(y_true, y_pred)
# evaluator.print_metrics()
# logs["metrics"] = metrics

# save_path = save_experiment_artifacts(
#     base_path=results_dir,
#     model=pipeline.model,
#     metrics_dict=metrics,
#     input_scaler=input_scaler,
#     output_scaler=output_scaler,
#     extra_files={
#         "train_test_loss.png": lambda p: plot_training_and_validation_loss(logs["train_loss"], logs["test_loss"], p),
#         "true_vs_pred.png": lambda p: plot_true_vs_pred(y_true, y_pred, p),
#     },
#     return_path=True
# )


# ─────────────────────────────────────────────────────
# Debugging
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
# INFERENCE TEST
# ─────────────────────────────────────────────────────
# # External evaluation on training set
# preds = model.predict(ds_train)
# targets = ds_train[:][1]
# mse = F.mse_loss(preds, targets)
# print(f"[External Eval] MSE: {mse:.6f}")


# y_pred = model_reloaded.predict(ds_test)
# y_pred = y_pred.detach().cpu().numpy()
# y_true = ds_test[:][1]
# y_true = y_true.detach().cpu().numpy()

# print(f"\n>>> Dataset: test")
# print(f"\n>>> Predictions shape: {y_pred.shape}")
# print(f">>> True values shape: {y_true.shape}")
# print(f">>> First 5 predictions: {y_pred[:5, :5]}")
# print(f">>> First 5 true values: {y_true[:5, :5]}")


# evaluator = RegressionEvaluator()
# metrics = evaluator(y_true, y_pred)
# evaluator.print_metrics()

# y_pred = model_reloaded.predict(ds_train)
# y_pred = y_pred.detach().cpu().numpy()
# y_true = ds_train[:][1]
# y_true = y_true.detach().cpu().numpy()

# print(f"\n>>> Dataset: train")
# print(f"\n>>> Predictions shape: {y_pred.shape}")
# print(f">>> True values shape: {y_true.shape}")
# print(f">>> First 5 predictions: {y_pred[:5, :5]}")
# print(f">>> First 5 true values: {y_true[:5, :5]}")

# evaluator = RegressionEvaluator()
# metrics = evaluator(y_true, y_pred)
# evaluator.print_metrics()

# sample_input = np.array([[4.0, 0.7]])  # AoA, Mach
# scaled_input = input_scaler_reloaded.transform(sample_input)
# input_tensor = torch.tensor(scaled_input, dtype=torch.float32, device=model_reloaded.device)

# with torch.no_grad():
#     prediction = model_reloaded.predict(input_tensor)

# print(f">>> Prediction shape: {prediction.shape}")
# print(f">>> First 5 values: {prediction[0, :5].detach().cpu().numpy()}")

# # Compare with nearest test sample
# X_test, Y_test = ds_test[:]
# nearest_idx = np.argmin(np.linalg.norm(
#     np.array(X_test) - np.array(sample_input), axis=1
# ))

# reference = Y_test[nearest_idx]

# print(f">>> Comparing with test sample at index {nearest_idx}")
# plot_true_vs_pred(reference, prediction[0].cpu().numpy())


# ─────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────

cr_info()
