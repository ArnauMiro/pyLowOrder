#!/usr/bin/env python

"""
Example: Training a GNS model on DLR airfoil data using pyLOM.

Updated to use modern pyLOM API:
  - Direct dict-based GNS instantiation
  - Optuna-compatible Pipeline execution

Author: Pablo Yeste
Date: 2025-07-28
"""

# ─────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import yaml
import hashlib
import datetime

from pyLOM.utils import raiseError
from pyLOM.NN.utils.config_loader_factory import load_yaml
from pyLOM.NN import Dataset, GNS, Pipeline, MinMaxScaler
from pyLOM.NN.optimizer import OptunaOptimizer
from pyLOM.NN.utils import RegressionEvaluator
from pyLOM.NN import DEVICE, set_seed
import pyLOM


# ─────────────────────────────────────────────────────
# UTILITIES: Plotting and Saving
# ─────────────────────────────────────────────────────

def plot_train_test_loss(train_loss, test_loss, path):
    """Plot training and validation loss and save."""
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    total_epochs = len(test_loss)
    iters_per_epoch = len(train_loss) // total_epochs
    plt.plot(np.arange(iters_per_epoch, len(train_loss) + 1, iters_per_epoch), test_loss, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(path, dpi=300)
    plt.close()


def true_vs_pred_plot(y_true, y_pred, path=None):
    """Plot true vs predicted values and optionally save to file."""
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
        plt.grid()
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300)
        plt.close()


def save_experiment(base_path, model, train_config, metrics_dict,
                    input_scaler=None, output_scaler=None, extra_files=None):
    """Save model, configs, metrics and optional scalers and figures."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    config_dict = model.config.asdict() if hasattr(model.config, 'asdict') else vars(model.config)
    yaml_path = base_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"model": config_dict, "training": train_config}, f)

    config_hash = hashlib.sha1(yaml_path.read_bytes()).hexdigest()
    timestamp = datetime.datetime.now().isoformat()
    with open(base_path / "meta.yaml", "w") as f:
        yaml.dump({"config_sha1": config_hash, "saved_at": timestamp}, f)

    model.save(base_path / "model.pth")

    if input_scaler:
        with open(base_path / "input_scaler.pkl", "wb") as f:
            pickle.dump(input_scaler, f)
    if output_scaler:
        with open(base_path / "output_scaler.pkl", "wb") as f:
            pickle.dump(output_scaler, f)

    with open(base_path / "metrics.yaml", "w") as f:
        yaml.dump(metrics_dict, f)

    if extra_files:
        for name, gen in extra_files.items():
            gen(base_path / name)


# ─────────────────────────────────────────────────────
# SETUP: Load config, seed, results path
# ─────────────────────────────────────────────────────
yaml_path = Path("../pyLowOrder/Examples/NN/configs/gns_config.yaml").absolute()
config_dict = load_yaml(yaml_path)

resudir = Path(config_dict["experiment"]["resudir"]).absolute()
resudir.mkdir(parents=True, exist_ok=True)

seed = config_dict["experiment"].get("seed", None)
device = config_dict["experiment"].get("device", DEVICE)
set_seed(seed, device)


# ─────────────────────────────────────────────────────
# LOAD DATASETS AND SCALERS
# ─────────────────────────────────────────────────────
input_scaler = MinMaxScaler()
output_scaler = None

dataset_kwargs = dict(
    field_names=["CP"],
    add_variables=True,
    add_mesh_coordinates=False,
    variables_names=["AoA", "Mach"],
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    squeeze_last_dim=False
)

td_train = Dataset.load(config_dict["datasets"]["train_ds"], **dataset_kwargs)
td_val   = Dataset.load(config_dict["datasets"]["val_ds"], **dataset_kwargs)
td_test  = Dataset.load(config_dict["datasets"]["test_ds"], **dataset_kwargs)


# ─────────────────────────────────────────────────────
# TRAIN OR OPTIMIZE MODEL
# ─────────────────────────────────────────────────────
mode = config_dict.get("experiment", {}).get("mode", "train")

if mode == "optuna":
    optimizer = OptunaOptimizer(
        optimization_params=config_dict["optuna"]["optimization_params"],
        n_trials=config_dict["optuna"]["study"].get("n_trials", 100),
        direction=config_dict["optuna"]["study"].get("direction", "minimize"),
        pruner=config_dict["optuna"]["study"].get("pruner"),
        save_dir=config_dict["optuna"]["study"].get("save_dir")
    )

    pipeline = Pipeline(
        train_dataset=td_train,
        valid_dataset=td_val,
        test_dataset=td_test,
        optimizer=optimizer,
        model_class=GNS,
    )
elif mode == "train":
    model = GNS(config=config_dict['model'])
    pipeline = Pipeline(
        train_dataset=td_train,
        valid_dataset=td_val,
        test_dataset=td_test,
        model=model,
        training_params=config_dict["fit"]
    )
else:
    raiseError(f"Unsupported mode '{mode}'. Use 'train' or 'optuna'.")

logs = pipeline.run()


# ─────────────────────────────────────────────────────
# EVALUATE AND SAVE EXPERIMENT
# ─────────────────────────────────────────────────────
predictions = pipeline.model.predict(td_test)
targets_eval = td_test[:][1]

evaluator = RegressionEvaluator()
evaluator(targets_eval, predictions)
evaluator.print_metrics()
logs["metrics"] = evaluator.metrics_dict

save_experiment(
    base_path=resudir,
    model=pipeline.model,
    train_config=config_dict["training"],
    metrics_dict=evaluator.metrics_dict,
    input_scaler=input_scaler,
    output_scaler=output_scaler,
    extra_files={
        "train_test_loss.png": lambda p: plot_train_test_loss(logs["train_loss"], logs["test_loss"], p),
        "true_vs_pred.png": lambda p: true_vs_pred_plot(targets_eval, predictions, p),
    }
)


# ─────────────────────────────────────────────────────
# INFERENCE TEST: Reload model and predict new input
# ─────────────────────────────────────────────────────
print("\n>>> Reloading model and input scaler from disk...")
model_reloaded = GNS.load(resudir / "model.pth")

scaler_path = resudir / "input_scaler.pkl"
if not scaler_path.exists():
    raise FileNotFoundError(f"Missing input scaler: {scaler_path}")
with open(scaler_path, "rb") as f:
    input_scaler_reloaded = pickle.load(f)

raw_input = np.array([[4.0, 0.7]])  # AoA, Mach
scaled_input = input_scaler_reloaded.transform(raw_input)
input_tensor = torch.tensor(scaled_input, dtype=torch.float32, device=model_reloaded.device)

with torch.no_grad():
    pred_dummy = model_reloaded.predict(input_tensor)

print(f">>> Prediction shape: {pred_dummy.shape}")
print(f">>> First 5 values: {pred_dummy[0, :5].detach().cpu().numpy()}")

# Optional: compare with nearest test sample
inputs_test, targets_test = td_test[:]
idx = np.argmin(np.linalg.norm(inputs_test - raw_input, axis=1))
reference = targets_test[idx]
print(f">>> Comparing with test sample at index {idx}")
true_vs_pred_plot(reference, pred_dummy[0].cpu().numpy())


# ─────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────
pyLOM.cr_info()
