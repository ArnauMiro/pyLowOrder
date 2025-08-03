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
from typing import Any, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import yaml
import hashlib
import datetime
from dataclasses import asdict

from pyLOM.NN import Dataset, GNS, Pipeline, MinMaxScaler
from pyLOM.NN.optimizer import OptunaOptimizer
from pyLOM.NN.utils import RegressionEvaluator
from pyLOM.NN.utils.config_serialization import load_yaml, serialize_config, deserialize_config
from pyLOM import cr_info


# ─────────────────────────────────────────────────────
# UTILITIES: PLOTTING AND SAVING
# ─────────────────────────────────────────────────────

def plot_training_and_validation_loss(train_loss: list[float],
                                      val_loss: list[float],
                                      save_path: Path) -> None:
    """
    Plot and save training and validation loss curves.

    Args:
        train_loss (list[float]): Training loss values per iteration.
        val_loss (list[float]): Validation loss values per epoch.
        save_path (Path): File path where the plot will be saved.
    """
    if not train_loss or not val_loss:
        raise ValueError("Both train_loss and val_loss must be non-empty.")

    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")

    num_epochs = len(val_loss)
    iters_per_epoch = len(train_loss) // num_epochs
    val_iters = np.arange(iters_per_epoch, len(train_loss) + 1, iters_per_epoch)
    plt.plot(val_iters, val_loss, label="Validation Loss")

    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_true_vs_pred(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      save_path: Path | None = None) -> None:
    """
    Plot predicted vs. true values.

    Args:
        y_true (np.ndarray): Ground-truth values.
        y_pred (np.ndarray): Predicted values.
        save_path (Path | None): Optional path to save the figure.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")

    num_outputs = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_outputs))

    for i in range(num_outputs):
        plt.subplot(num_outputs, 1, i + 1)
        plt.scatter(y_true[:, i], y_pred[:, i], s=1, alpha=0.5)
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.grid()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()



def save_experiment_artifacts(base_path: Path,
                               model: Any,
                               metrics_dict: Dict[str, float],
                               input_scaler: Any = None,
                               output_scaler: Any = None,
                               extra_files: Dict[str, Any] | None = None) -> None:
    """
    Save all relevant experiment artifacts to disk.

    Args:
        base_path (Path): Root directory where to store outputs.
        model (Any): Trained model with `.config` and `.save()` interface.
        metrics_dict (Dict[str, float]): Dictionary of evaluation metrics.
        input_scaler (Any, optional): Input scaler object to serialize.
        output_scaler (Any, optional): Output scaler object to serialize.
        extra_files (Dict[str, Callable], optional): 
            Dictionary mapping filenames to functions that accept a Path.
    """
    base_path.mkdir(parents=True, exist_ok=True)

    yaml_config_path = base_path / "config.yaml"
    yaml.safe_dump({
        "model": serialize_config(asdict(model.config)),
        "fit": serialize_config(asdict(model.last_training_config)),
    }, yaml_config_path.open("w"))

    config_hash = hashlib.sha1(yaml_config_path.read_bytes()).hexdigest()
    timestamp = datetime.datetime.now().isoformat()
    yaml.safe_dump({
        "config_sha1": config_hash,
        "saved_at": timestamp
    }, (base_path / "meta.yaml").open("w"))

    model.save(base_path / "model.pth")

    if input_scaler:
        with open(base_path / "input_scaler.pkl", "wb") as f:
            pickle.dump(input_scaler, f)
    if output_scaler:
        with open(base_path / "output_scaler.pkl", "wb") as f:
            pickle.dump(output_scaler, f)

    with open(base_path / "metrics.yaml", "w") as f:
        yaml.safe_dump(metrics_dict, f)

    if extra_files:
        for filename, generator_fn in extra_files.items():
            if not callable(generator_fn):
                raise TypeError(f"Expected a callable for file '{filename}'")
            generator_fn(base_path / filename)



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
    optimizer = OptunaOptimizer(
        optimization_params=optuna_params["optimization_params"],
        n_trials=optuna_params["n_trials"],
        direction=optuna_params["direction"],
        pruner=optuna_params["pruner"],
        sampler=optuna_params["sampler"],
        seed=optuna_params.get("seed", 42),
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

logs = pipeline.run()


# ─────────────────────────────────────────────────────
# EVALUATION AND EXPERIMENT SAVING
# ─────────────────────────────────────────────────────

y_pred = pipeline.model.predict(ds_test)
y_true = ds_test[:][1]

evaluator = RegressionEvaluator()
evaluator(y_true, y_pred)
evaluator.print_metrics()
logs["metrics"] = evaluator.metrics_dict

save_experiment_artifacts(
    base_path=results_dir,
    model=pipeline.model,
    metrics_dict=evaluator.metrics_dict,
    input_scaler=input_scaler,
    output_scaler=output_scaler,
    extra_files={
        "train_test_loss.png": lambda p: plot_training_and_validation_loss(logs["train_loss"], logs["test_loss"], p),
        "true_vs_pred.png": lambda p: plot_true_vs_pred(y_true, y_pred, p),
    }
)


# ─────────────────────────────────────────────────────
# INFERENCE TEST
# ─────────────────────────────────────────────────────

print("\n>>> Reloading model and input scaler from disk...")
model_reloaded = GNS.load(results_dir / "model.pth")

with open(results_dir / "input_scaler.pkl", "rb") as f:
    input_scaler_reloaded = pickle.load(f)

sample_input = np.array([[4.0, 0.7]])  # AoA, Mach
scaled_input = input_scaler_reloaded.transform(sample_input)
input_tensor = torch.tensor(scaled_input, dtype=torch.float32, device=model_reloaded.device)

with torch.no_grad():
    prediction = model_reloaded.predict(input_tensor)

print(f">>> Prediction shape: {prediction.shape}")
print(f">>> First 5 values: {prediction[0, :5].detach().cpu().numpy()}")

# Compare with nearest test sample
X_test, Y_test = ds_test[:]
nearest_idx = np.argmin(np.linalg.norm(X_test - sample_input, axis=1))
reference = Y_test[nearest_idx]

print(f">>> Comparing with test sample at index {nearest_idx}")
plot_true_vs_pred(reference, prediction[0].cpu().numpy())


# ─────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────

cr_info()
