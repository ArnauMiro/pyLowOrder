#!/usr/bin/env python

"""
Example: Training a GNS model on DLR airfoil data using pyLOM.

This script demonstrates the key steps required to:
1. Load configuration from a YAML file
2. Initialize scalers and datasets
3. Build and train a GNS model or run Optuna optimization
4. Evaluate the model and visualize results
5. Save the model and associated artifacts
6. Perform a dummy inference

Author: Pablo Yeste
Date: 2025-07-25
"""

# ─────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import yaml
import hashlib
import datetime

from pyLOM.utils import load_yaml, build_GNS_config
from pyLOM.NN import Dataset, GNS, Pipeline, MinMaxScaler
from pyLOM.NN.optimizer import OptunaOptimizer
from pyLOM.NN.utils import RegressionEvaluator
import pyLOM


# ─────────────────────────────────────────────────────
# PLOTTING FUNCTIONS (moved up for clarity)
# ─────────────────────────────────────────────────────
def plot_train_test_loss(train_loss, test_loss, path):
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    total_epochs = len(test_loss)
    iters_per_epoch = len(train_loss) // total_epochs
    plt.plot(np.arange(iters_per_epoch, len(train_loss) + 1, iters_per_epoch), test_loss, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid()
    plt.legend()
    plt.savefig(path, dpi=300)
    plt.close()

def true_vs_pred_plot(y_true, y_pred, path=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_outputs = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_outputs))
    for i in range(num_outputs):
        plt.subplot(num_outputs, 1, i + 1)
        plt.scatter(y_true[:, i], y_pred[:, i], s=1, alpha=0.5)
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                 [y_true[:, i].min(), y_true[:, i].max()], 'r--')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Output component {i+1}")
        plt.grid()
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300)
        plt.close()
    else:
        plt.show()


# ─────────────────────────────────────────────────────
# SAVE EXPERIMENT FUNCTION (for now kept local)
# ─────────────────────────────────────────────────────
def save_experiment(
    base_path: Path,
    model,
    train_config: dict,
    metrics_dict: dict,
    input_scaler=None,
    output_scaler=None,
    extra_files: dict = None
):
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Save YAML config from model.config
    config_dict = model.config.asdict() if hasattr(model.config, 'asdict') else vars(model.config)
    yaml_path = base_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"model": config_dict, "training": train_config}, f)

    # Compute SHA-1 and timestamp
    config_hash = hashlib.sha1(yaml_path.read_bytes()).hexdigest()
    timestamp = datetime.datetime.now().isoformat()

    # Write metadata
    with open(base_path / "meta.yaml", "w") as f:
        yaml.dump({
            "config_sha1": config_hash,
            "saved_at": timestamp
        }, f)

    # Save model
    model.save(base_path / "model.pth")

    # Save scalers
    if input_scaler is not None:
        with open(base_path / "input_scaler.pkl", "wb") as f:
            pickle.dump(input_scaler, f)

    if output_scaler is not None:
        with open(base_path / "output_scaler.pkl", "wb") as f:
            pickle.dump(output_scaler, f)

    # Save metrics
    with open(base_path / "metrics.yaml", "w") as f:
        yaml.dump(metrics_dict, f)

    # Save extra files
    if extra_files:
        for name, generator in extra_files.items():
            generator(base_path / name)


# ─────────────────────────────────────────────────────
# STEP 1 — Load YAML Configuration
# ─────────────────────────────────────────────────────
yaml_path = "./configs/gns_config.yaml"
config = load_yaml(yaml_path)
resudir = Path(config["execution"]["resudir"])
resudir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────
# STEP 2 — Setup Device and Seed
# ─────────────────────────────────────────────────────
device = config.get("experiment", {}).get("device", "cuda")
seed = config.get("experiment", {}).get("seed", None)

if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────
# STEP 3 — Initialize Scalers and Load Datasets
# ─────────────────────────────────────────────────────
input_scaler = MinMaxScaler()
output_scaler = None

kwargs = dict(
    field_names=["CP"],
    add_variables=True,
    add_mesh_coordinates=False,
    variables_names=["AoA", "Mach"],
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    squeeze_last_dim=False
)

td_train = Dataset.load(config["datasets"]["train_ds"], **kwargs)
td_val   = Dataset.load(config["datasets"]["val_ds"], **kwargs)
td_test  = Dataset.load(config["datasets"]["test_ds"], **kwargs)

# ─────────────────────────────────────────────────────
# BLOCK A — Train GNS from config
# ─────────────────────────────────────────────────────
gns_config = build_GNS_config(config)
model = GNS(gns_config)
pipeline = Pipeline(
    train_dataset=td_train,
    valid_dataset=td_val,
    test_dataset=td_test,
    model=model,
    training_params=config["training"]
)
logs = pipeline.run()

# ─────────────────────────────────────────────────────
# BLOCK B — Optimize GNS with Optuna (optional)
# ─────────────────────────────────────────────────────
# Uncomment this block to run Optuna hyperparameter search
# optimizer = OptunaOptimizer(
#     search_space=config["optuna"]["search_space"],
#     study_params=config["optuna"]["study"],
#     graph_path=config["optuna"]["graph_path"]
# )
# pipeline = Pipeline(
#     train_dataset=td_train,
#     valid_dataset=td_val,
#     test_dataset=td_test,
#     optimizer=optimizer,
#     model_class=GNS,
# )
# logs = pipeline.run()

# ─────────────────────────────────────────────────────
# STEP 5 — Evaluate and Save
# ─────────────────────────────────────────────────────
preds = pipeline.model.predict(td_test)
targets = td_test[:][1]

evaluator = RegressionEvaluator()
evaluator(targets, preds)
evaluator.print_metrics()
logs["metrics"] = evaluator.metrics_dict

save_experiment(
    base_path=resudir,
    model=pipeline.model,
    train_config=config["training"],
    metrics_dict=evaluator.metrics_dict,
    input_scaler=input_scaler,
    output_scaler=output_scaler,
    extra_files={
        "train_test_loss.png": lambda p: plot_train_test_loss(logs["train_loss"], logs["test_loss"], p),
        "true_vs_pred.png": lambda p: true_vs_pred_plot(targets, preds, p),
    }
)

# ─────────────────────────────────────────────────────
# STEP 6 — Reload and Inference
# ─────────────────────────────────────────────────────
print("\n>>> Reloading model and scaler from disk...")
model_reloaded = GNS.load(resudir / "model.pth")

scaler_path = resudir / "input_scaler.pkl"
if scaler_path.exists():
    with open(scaler_path, "rb") as f:
        input_scaler_reloaded = pickle.load(f)
else:
    raise FileNotFoundError(f"{scaler_path} not found.")

raw_input = np.array([[4.0, 0.7]])
scaled_input = input_scaler_reloaded.transform(raw_input)
input_tensor = torch.tensor(scaled_input, dtype=torch.float32, device=model_reloaded.device)

with torch.no_grad():
    pred_dummy = model_reloaded.predict(input_tensor)

print(f">>> Prediction shape: {pred_dummy.shape}")
print(f">>> First 5 values: {pred_dummy[0, :5].detach().cpu().numpy()}")

# ─────────────────────────────────────────────────────
# STEP 7 — Optional Comparison
# ─────────────────────────────────────────────────────
if 'td_test' in globals():
    inputs, targets = td_test[:]
    idx = np.argmin(np.linalg.norm(inputs - raw_input, axis=1))
    reference = targets[idx]
    print(f">>> Comparing with test sample at index {idx}")
    true_vs_pred_plot(reference, pred_dummy[0].cpu().numpy())


# ─────────────────────────────────────────────────────
# END
# ─────────────────────────────────────────────────────
pyLOM.cr_info()
