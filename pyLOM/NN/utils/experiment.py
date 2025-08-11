"""
Utilites for running and saving experiments with NN modules in pyLOM.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import yaml
import hashlib
import datetime
from dataclasses import asdict
import getpass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from dataclasses import asdict
import datetime, getpass, hashlib, json, pickle, torch, yaml

from ... import pprint
from ...utils import get_git_commit, raiseError
from ...utils.config_resolvers import to_native


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
    Plot predicted vs. true values with sensible figsize for many outputs.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")

    num_outputs = y_true.shape[1]

    # Limit figure size growth
    height_per_plot = 3
    max_height = 20
    fig_height = min(height_per_plot * num_outputs, max_height)

    plt.figure(figsize=(8, fig_height))

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


def save_experiment_artifacts(
    base_path: Path,
    model: any,
    metrics_dict: dict[str, float],
    inputs_scaler: any = None,
    outputs_scaler: any = None,
    extra_files: dict[str, any] | None = None,
    return_path: bool = False,
) -> Path | None:
    """
    Save model checkpoint + DTO configs + provenance + scalers + metrics.

    Notes
    -----
    - The written config.yaml mirrors what the checkpoint stores (DTOs + provenance).
    - A stable SHA256 is computed from a canonical JSON dump of that config document.
    """
    # 1) Decide output directory (timestamped subfolder if 'base_path' is a dir)
    if base_path.is_dir() or not base_path.suffix:
        timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
        model_name = getattr(model, "__class__", type("X",(object,),{})).__name__
        out_dir = base_path / f"{timestamp}_{model_name}"
    else:
        out_dir = base_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Build the DTO/provenance document we’ll save AND hash
    model_cfg_dict  = asdict(model.model_config)
    train_cfg_dict  = asdict(model.last_training_config) if getattr(model, "last_training_config", None) else None
    prov_dict       = {
        "graph_spec": asdict(model.graph_spec),
        "graph_fingerprint": getattr(model, "graph_fingerprint", None),
    }
    config_doc = {
        "model": model_cfg_dict,
        "training": train_cfg_dict,
        "provenance": prov_dict,
    }

    # 3) Stable hash from canonical JSON (independent of YAML formatting)
    canonical = json.dumps(config_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config_sha256 = hashlib.sha256(canonical).hexdigest()

    # 4) Write config.yaml (human‑readable) and meta.yaml
    yaml_config_path = out_dir / "config.yaml"
    with yaml_config_path.open("w") as f:
        yaml.safe_dump(config_doc, f, sort_keys=False)

    meta_info = {
        "saved_at": datetime.datetime.now().isoformat(),
        "config_sha256": config_sha256,
        "torch_version": str(torch.__version__),
        "user": getpass.getuser(),
        "git_commit": get_git_commit(),
    }
    meta_info = to_native(meta_info)  # Convert to native types
    with (out_dir / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta_info, f, sort_keys=False)

    # 5) Save model checkpoint (contains DTOs + provenance + states)
    model.save(out_dir / "model.pth")

    # 6) Save scalers (via their own API)
    # Use consistent, pluralized names and JSON extension.
    if inputs_scaler is not None:
        if not getattr(inputs_scaler, "is_fitted", True):
            raiseError("inputs_scaler must be fitted before saving.")
        inputs_scaler.save(str(out_dir / "inputs_scaler.json"))
        if hasattr(inputs_scaler, "save"):
            inputs_scaler.save(str(out_dir / "inputs_scaler.json"))
        else:
            raiseError("inputs_scaler does not implement a .save(filepath) method.")

    if outputs_scaler is not None:
        if hasattr(outputs_scaler, "save"):
            outputs_scaler.save(str(out_dir / "outputs_scaler.json"))
        else:
            raiseError("outputs_scaler does not implement a .save(filepath) method.")

    # 7) Save metrics (YAML with numpy→native conversion)
    native_metrics = to_native(metrics_dict)
    with (out_dir / "metrics.yaml").open("w") as f:
        yaml.safe_dump(native_metrics, f, sort_keys=False)

    # 8) Extra files
    if extra_files:
        for filename, generator_fn in extra_files.items():
            if not callable(generator_fn):
                raise TypeError(f"Expected a callable for file '{filename}'")
            generator_fn(out_dir / filename)

    pprint(0, f"Experiment artifacts saved to: {out_dir}")
    return out_dir if return_path else None


def evaluate_dataset_with_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    Evaluate predictions against targets using standard regression metrics.
    
    Args:
        preds (np.ndarray): Model predictions. Shape: [B, N, F] or [B, N].
        targets (np.ndarray): Ground truth values. Same shape as `preds`.

    Returns:
        dict: Dictionary of regression metrics.
    """
    # Flatten for per-sample evaluation
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    assert preds_flat.shape == targets_flat.shape, (
        f"Shape mismatch: preds {preds_flat.shape}, targets {targets_flat.shape}"
    )

    # Standard metrics
    mse = mean_squared_error(targets_flat, preds_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_flat, preds_flat)
    r2 = r2_score(targets_flat, preds_flat)

    # Relative errors
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.abs((targets_flat - preds_flat) / targets_flat)
        relative_errors = relative_errors[np.isfinite(relative_errors)]

    mre = np.mean(relative_errors) * 100 if len(relative_errors) > 0 else np.nan

    # Absolute error quantiles
    ae = np.abs(targets_flat - preds_flat)
    ae_95 = np.percentile(ae, 95)
    ae_99 = np.percentile(ae, 99)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mre": mre,
        "ae_95": ae_95,
        "ae_99": ae_99,
        "r2": r2,
        "l2_error": np.linalg.norm(preds_flat - targets_flat) / np.linalg.norm(targets_flat)
    }
