"""
Utilites for running and saving experiments with NN modules in pyLOM.
"""

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
import getpass

from pyLOM.NN.utils.config_serialization import serialize_config
from pyLOM import pprint
from pyLOM.utils import get_git_commit


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



def _convert_numpy_to_native(obj):
    """
    Recursively convert NumPy scalar types in a structure to native Python types.
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_numpy_to_native(v) for v in obj)
    elif hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return obj
    else:
        return obj


def save_experiment_artifacts(base_path: Path,
                               model: Any,
                               metrics_dict: Dict[str, float],
                               input_scaler: Any = None,
                               output_scaler: Any = None,
                               extra_files: Dict[str, Any] | None = None,
                               return_path: bool = False) -> None:
    """
    Save all relevant experiment artifacts to disk.
    If base_path is a directory with no specific subfolder, a timestamped folder will be created automatically.
    Args:
        base_path (Path): Base path where artifacts will be saved.
        model (Any): The trained model instance.
        metrics_dict (Dict[str, float]): Dictionary of evaluation metrics.
        input_scaler (Any, optional): Input scaler instance (e.g., MinMaxScaler).
        output_scaler (Any, optional): Output scaler instance (e.g., MinMaxScaler).
        extra_files (Dict[str, Any], optional): Additional files to save, with filename as key and a callable generator function as value.
        return_path (bool): If True, return the path where artifacts were saved.
    Returns:
        base_path (Path): The path where artifacts were saved, if return_path is True.
    Raises:
        TypeError: If extra_files values are not callable.
    """
    # Ensure base_path is a directory, generate subfolder if needed
    if base_path.is_dir() or not base_path.suffix:
        timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
        model_name = getattr(model.config, "model_name", model.__class__.__name__)
        base_path = base_path / f"{timestamp}_{model_name}"
    
    base_path.mkdir(parents=True, exist_ok=True)

    # Save config.yaml
    yaml_config_path = base_path / "config.yaml"
    yaml.safe_dump({
        "model": serialize_config(asdict(model.config)),
        "training": serialize_config(asdict(model.last_training_config)),
    }, yaml_config_path.open("w"))

    # Save meta.yaml with hash and timestamp
    config_hash = hashlib.sha1(yaml_config_path.read_bytes()).hexdigest()
    meta_info = {   
        "saved_at": datetime.datetime.now().isoformat(),
        "config_sha1": config_hash,
        "torch_version": torch.__version__,
        "user": getpass.getuser(),
        "git_commit": get_git_commit(),
    }
    yaml.safe_dump(meta_info, (base_path / "meta.yaml").open("w"))

    # Save model and scalers
    model.save(base_path / "model.pth")

    if input_scaler:
        with open(base_path / "input_scaler.pkl", "wb") as f:
            pickle.dump(input_scaler, f)
    if output_scaler:
        with open(base_path / "output_scaler.pkl", "wb") as f:
            pickle.dump(output_scaler, f)

    # Save evaluation metrics
    native_metrics = _convert_numpy_to_native(metrics_dict)
    with open(base_path / "metrics.yaml", "w") as f:
        yaml.safe_dump(native_metrics, f)

    # Save any extra files
    if extra_files:
        for filename, generator_fn in extra_files.items():
            if not callable(generator_fn):
                raise TypeError(f"Expected a callable for file '{filename}'")
            generator_fn(base_path / filename)

    pprint(0, f"Experiment artifacts saved to: {base_path}")

    if return_path:
        return base_path