"""
Utilites for running and saving experiments with NN modules in pyLOM.
"""

from pathlib import Path
import os
import datetime
import getpass
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
    Literal,
)

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import pyLOM

from ... import pprint
from ...utils import get_git_commit, raiseError
from ...utils.config_resolvers import to_native


if TYPE_CHECKING:
    from pyLOM import Mesh
    from pyLOM.partition_table import PartitionTable


ArrayLike = Union[np.ndarray, "torch.Tensor"]
MetricFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ParaviewExportConfig:
    """Configuration for exporting fields to ParaView-friendly vtk.hdf files."""

    mesh: "Mesh"
    cell_order: np.ndarray
    partition_table: "PartitionTable"
    instants: Sequence[int] | np.ndarray
    times: Sequence[float] | np.ndarray
    output_dir: Path
    base_name: str = "predictions"
    mode: Literal["single", "per_snapshot"] = "single"
    snapshot_metadata: Mapping[str, ArrayLike] | None = None
    extra_cell_fields: Mapping[str, ArrayLike] | None = None


# ─────────────────────────────────────────────────────
# UTILITIES: ARRAY HANDLING
# ─────────────────────────────────────────────────────

def to_numpy(value: ArrayLike | Sequence[Any], *, dtype: np.dtype | type | None = None,
             copy: bool = False) -> np.ndarray:
    """Convert tensors or array-likes to a NumPy array with optional dtype/copy.

    Parameters
    ----------
    value : array-like or torch.Tensor
        Input object to convert.
    dtype : numpy dtype or type, optional
        If provided, cast the resulting array to this dtype.
    copy : bool, default False
        Force copying the data. When ``False`` NumPy may reuse the underlying
        memory if possible.
    """

    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    if dtype is not None:
        array = array.astype(dtype, copy=False)

    if copy:
        array = np.array(array, dtype=array.dtype, copy=True)

    return array


# ─────────────────────────────────────────────────────
# PARAVIEW EXPORT UTILITIES
# ─────────────────────────────────────────────────────


def _ensure_sample_major(
    array: ArrayLike,
    ncells: int,
    *,
    nsnaps_hint: int | None = None,
) -> np.ndarray:
    """Return array shaped as (nsnaps, ncells, nfeat) with optional broadcasting."""

    arr = to_numpy(array)
    arr = np.asarray(arr)

    if arr.ndim == 1:
        if arr.size == ncells:
            arr = arr[None, :, None]
        elif nsnaps_hint is not None and arr.size == nsnaps_hint:
            arr = arr[:, None, None]
        else:
            raise ValueError("1D arrays must match either ncells or nsnaps length")
    elif arr.ndim == 2:
        if arr.shape[1] == ncells:
            arr = arr[:, :, None]
        elif arr.shape[0] == ncells:
            arr = arr.T[:, :, None]
        else:
            raise ValueError("2D arrays must include ncells in one axis")
    elif arr.ndim == 3:
        if arr.shape[1] == ncells:
            pass
        elif arr.shape[0] == ncells:
            arr = arr.swapaxes(0, 1)
        else:
            raise ValueError("3D arrays must include ncells in axis 0 or 1")
    else:
        raise ValueError("Unsupported array rank for sample-major conversion")

    if nsnaps_hint is not None:
        if arr.shape[0] == 1 and nsnaps_hint > 1:
            arr = np.broadcast_to(arr, (nsnaps_hint, arr.shape[1], arr.shape[2]))
        elif arr.shape[0] not in (nsnaps_hint, 1):
            raise ValueError("Snapshot dimension mismatch with provided hint")

    return arr.astype(np.float64, copy=False)


def _sample_to_cell_major(arr: np.ndarray) -> np.ndarray:
    """Swap axes to obtain (ncells, nsnaps, nfeat) layout."""

    return arr.swapaxes(0, 1)


def _reshape_for_dataset(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return flat array (ncomp * ncells, nsnaps) and component count."""

    ncells, nsnaps, nfeat = arr.shape
    data = np.moveaxis(arr, 2, 0).reshape(nfeat * ncells, nsnaps)
    return data, nfeat


def _build_dataset(
    config: ParaviewExportConfig,
    cell_fields: Mapping[str, np.ndarray],
    metadata: Mapping[str, ArrayLike] | None,
) -> pyLOM.Dataset:
    """Create a pyLOM Dataset with provided cell fields and snapshot metadata."""

    if not cell_fields:
        raise ValueError("At least one field must be supplied for ParaView export")

    sample_field = next(iter(cell_fields.values()))
    _, nsnaps, _ = sample_field.shape

    vars_dict: Dict[str, Dict[str, Any]] = {}
    if metadata:
        for name, values in metadata.items():
            values_np = to_numpy(values)
            values_np = np.asarray(values_np)
            if values_np.ndim == 1:
                if values_np.size not in (nsnaps, 1):
                    raise ValueError(f"Metadata '{name}' must have length 1 or nsnaps")
                if values_np.size == 1 and nsnaps > 1:
                    values_np = np.broadcast_to(values_np, (nsnaps,))
                vars_dict[name] = {"idim": 0, "value": values_np.astype(np.float64, copy=False)}
            elif values_np.ndim == 2:
                if values_np.shape[0] not in (nsnaps, 1):
                    raise ValueError(f"Metadata '{name}' must align with snapshot count")
                if values_np.shape[0] == 1 and nsnaps > 1:
                    values_np = np.broadcast_to(values_np, (nsnaps, values_np.shape[1]))
                vars_dict[name] = {
                    "idim": values_np.shape[1],
                    "value": values_np.astype(np.float64, copy=False),
                }
            else:
                raise ValueError(f"Metadata '{name}' must be 1D or 2D array")

    dataset = pyLOM.Dataset(
        xyz=config.mesh.xyzc,
        ptable=config.partition_table,
        order=config.cell_order,
        point=False,
        vars=vars_dict,
    )

    for field_name, field_values in cell_fields.items():
        flat_values, ncomp = _reshape_for_dataset(field_values)
        dataset.add_field(field_name, ncomp, flat_values)

    return dataset


def _format_metadata_suffix(metadata: Mapping[str, ArrayLike] | None) -> str:
    if not metadata:
        return ""

    parts: List[str] = []
    for name, values in metadata.items():
        arr = np.asarray(to_numpy(values), dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        value = float(arr[0])
        formatted = format(value, ".6g").replace("-", "m").replace(".", "p")
        parts.append(f"{name}_{formatted}")

    return "__".join(parts)


def export_predictions_to_paraview(
    y_pred: ArrayLike,
    y_true: ArrayLike,
    metrics: Iterable[tuple[str, MetricFn]] | None,
    *,
    config: ParaviewExportConfig,
) -> List[Path]:
    """
    Export predictions/targets and derived metrics to ParaView (vtk.hdf) files.

    All tensors must be descaled prior to calling this function. Fields are
    stored as cell-centered data. Snapshot metadata (inputs, conditions, etc.)
    can be provided via ``config.snapshot_metadata``.
    """

    ncells = config.mesh.ncells

    pred_samples = _ensure_sample_major(y_pred, ncells)
    true_samples = _ensure_sample_major(y_true, ncells)

    if pred_samples.shape != true_samples.shape:
        raise ValueError("Predictions and targets must share the same shape")

    nsnaps = pred_samples.shape[0]

    instants = np.asarray(config.instants, dtype=np.int32)
    times = np.asarray(config.times, dtype=np.float64)
    if instants.shape[0] != nsnaps or times.shape[0] != nsnaps:
        raise ValueError("Instants/times length must match number of snapshots")

    cell_fields: Dict[str, np.ndarray] = {
        "target": _sample_to_cell_major(true_samples),
    }

    metric_items = list(metrics) if metrics else [("prediction", lambda yp, yt: yp)]
    for name, fn in metric_items:
        metric_result = fn(pred_samples, true_samples)
        metric_samples = _ensure_sample_major(metric_result, ncells, nsnaps_hint=nsnaps)
        if metric_samples.shape != pred_samples.shape:
            raise ValueError(f"Metric '{name}' must return the same shape as inputs")
        if name in cell_fields:
            raise ValueError(f"Duplicate field name detected: {name}")
        cell_fields[name] = _sample_to_cell_major(metric_samples)

    metadata_arrays: Dict[str, np.ndarray] = {}
    if config.snapshot_metadata:
        for name, values in config.snapshot_metadata.items():
            arr = np.asarray(to_numpy(values), dtype=float).reshape(-1)
            if arr.size not in (1, nsnaps):
                raise ValueError(f"Metadata '{name}' must have length 1 or nsnaps")
            if arr.size == 1 and nsnaps > 1:
                arr = np.broadcast_to(arr, nsnaps)
            metadata_arrays[name] = arr

            constant = np.broadcast_to(arr.reshape(1, -1), (ncells, nsnaps))
            cell_fields[f"meta_{name}"] = constant[:, :, None]

    if config.extra_cell_fields:
        for name, values in config.extra_cell_fields.items():
            if name in cell_fields:
                raise ValueError(f"Duplicate field name detected: {name}")
            extra_samples = _ensure_sample_major(values, ncells, nsnaps_hint=nsnaps)
            cell_fields[name] = _sample_to_cell_major(extra_samples)

    dataset = _build_dataset(config, cell_fields, config.snapshot_metadata)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    mode = config.mode
    casestr_base = config.base_name or "predictions"

    if mode == "single":
        pyLOM.io.pv_writer(
            Mesh=config.mesh,
            Dataset=dataset,
            casestr=casestr_base,
            basedir=str(output_dir),
            instants=instants,
            times=times,
            vars=list(dataset.fields.keys()),
            fmt="vtkh5",
            mode="w",
        )
        written.append(output_dir / f"{casestr_base}.vtk.hdf")
    elif mode == "per_snapshot":
        for idx in range(nsnaps):
            slice_fields = {
                name: values[:, idx : idx + 1, :]
                for name, values in cell_fields.items()
            }
            slice_metadata = (
                {
                    name: metadata_arrays[name][idx : idx + 1]
                    for name in metadata_arrays
                }
                if config.snapshot_metadata
                else None
            )
            slice_dataset = _build_dataset(config, slice_fields, slice_metadata)
            suffix = _format_metadata_suffix(slice_metadata)
            if suffix:
                casestr = f"{casestr_base}_{idx:04d}__{suffix}"
            else:
                casestr = f"{casestr_base}_{idx:04d}"

            instants_slice = np.asarray([0], dtype=np.int32)
            times_slice = np.asarray([times[idx]], dtype=np.float64)
            pyLOM.io.pv_writer(
                Mesh=config.mesh,
                Dataset=slice_dataset,
                casestr=casestr,
                basedir=str(output_dir),
                instants=instants_slice,
                times=times_slice,
                vars=list(slice_dataset.fields.keys()),
                fmt="vtkh5",
                mode="w",
            )
            written.append(output_dir / f"{casestr}.vtk.hdf")
    else:
        raise ValueError("mode must be either 'single' or 'per_snapshot'")

    return written


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


# ----------------------------- core metrics ----------------------------- #

def compute_regression_metrics(
    preds: ArrayLike,
    targets: ArrayLike,
    *,
    percentiles: Sequence[float] = (95.0, 99.0),
    eps: float = 1e-12,
    finite_only: bool = True,
) -> Dict[str, float]:
    """
    Compute standard regression metrics on predictions vs. targets.

    Parameters
    ----------
    preds, targets : array-like
        Predictions and ground truth. Accept numpy arrays or torch tensors.
        Shapes can be (B, N, F), (B, N) or flat; data will be flattened.
    percentiles : sequence of float, optional
        Percentiles (0-100) of absolute error to report as keys 'ae_p{p}'.
    eps : float, optional
        Small constant to avoid division by zero in relative errors.
    finite_only : bool, optional
        If True, drop any non-finite pairs before computing metrics.

    Returns
    -------
    Dict[str, float]
        Metrics including: mse, rmse, mae, mre (in %), r2, bias, pearson_r,
        l2_error, and absolute error percentiles 'ae_p{p}'.
    """
    def _to_numpy_flat(a: ArrayLike) -> np.ndarray:
        return np.asarray(to_numpy(a), dtype=float).reshape(-1)

    y_pred = _to_numpy_flat(preds)
    y_true = _to_numpy_flat(targets)

    if finite_only:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]

    if y_true.size == 0:
        raise ValueError("No valid samples to evaluate after filtering.")

    err = y_pred - y_true
    abs_err = np.abs(err)

    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_err))
    bias = float(np.mean(err))

    # R^2
    denom = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - np.sum((y_true - y_pred)**2) / denom) if denom > 0 else np.nan

    # Pearson r
    std_true, std_pred = float(np.std(y_true)), float(np.std(y_pred))
    pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1]) if (std_true > 0 and std_pred > 0) else np.nan

    # Mean Relative Error (%)
    rel = np.abs((y_true - y_pred) / (np.abs(y_true) + eps))
    rel = rel[np.isfinite(rel)]
    mre = float(np.mean(rel) * 100.0) if rel.size > 0 else np.nan

    # L2 relative error
    l2_num = float(np.linalg.norm(y_pred - y_true))
    l2_den = float(np.linalg.norm(y_true)) + eps
    l2_error = l2_num / l2_den

    out: Dict[str, float] = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mre": mre,
        "r2": r2,
        "bias": bias,
        "pearson_r": pearson_r,
        "l2_error": l2_error,
        "n": float(y_true.size),
    }
    for p in percentiles:
        out[f"ae_p{int(p)}"] = float(np.percentile(abs_err, p))
    return out


# --------------------------- model evaluation --------------------------- #

def evaluate_model(model, training_params, dataset, evaluators) -> Tuple[Dict[str, float], Dict[str, ArrayLike]]:
    """
    Run model prediction on a dataset, inverse-scale, and compute evaluator metrics.

    Relies on scalers attached to `dataset` (dataset.inputs_scaler / dataset.outputs_scaler).
    Ensures y_true and y_pred are returned as column vectors with shape (N*, 1).

    Returns
    -------
    metrics : dict
        Aggregated metrics from `evaluators`.
    payload : dict
        Dictionary with raw arrays:
          - x_true : inputs (possibly None)
          - y_true : (N*, 1) numpy array
          - y_pred : (N*, 1) numpy array
    """
    def _as_column(arr: ArrayLike) -> np.ndarray:
        # Convert to numpy and ensure shape (N*, 1)
        return to_numpy(arr).reshape(-1, 1)

    # --- Predict ---
    print(f"Evaluating model: {model.__class__.__name__}")
    y_pred = model.predict(dataset, **training_params)
    print(f"Raw predictions shape: {getattr(y_pred, 'shape', None)}")

    # Inverse-scale predictions using dataset's attached output scaler
    y_pred = dataset.inverse_scale_outputs(y_pred)

    # --- Ground truth ---
    batch = dataset[:]
    if isinstance(batch, tuple):
        x_true, y_true = batch
        x_true = dataset.inverse_scale_inputs(x_true)
    else:
        x_true, y_true = None, batch

    y_true = dataset.inverse_scale_outputs(y_true)

    # --- Normalize to (N*, 1) ---
    y_true_col = _as_column(y_true)
    y_pred_col = _as_column(y_pred)

    # --- External evaluators ---
    metrics: Dict[str, float] = {}
    for ev in evaluators:
        metrics.update(ev(y_true_col, y_pred_col))
        if hasattr(ev, "print_metrics"):
            ev.print_metrics()

    payload = {"x_true": x_true, "y_true": y_true_col, "y_pred": y_pred_col}
    return metrics, payload


# ------------------------------ plotting ------------------------------- #

def plot_true_vs_pred(
    ax: plt.Axes,
    y_true_col: ArrayLike,
    y_pred_col: ArrayLike,
    title: str = "True vs. Predicted",
    *,
    identity_line: bool = True,
    fit_line: bool = True,
    equal_limits: bool = True,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    s: float = 8.0,
    alpha: float = 0.5,
    show_metrics_on_figure: bool = True,
    metrics: Optional[Dict[str, float]] = None,
    metrics_to_show: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Plot a scatter of y_true vs. y_pred with optional identity/regression lines and
    optionally annotate metrics on the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    y_true_col, y_pred_col : array-like
        Column vectors (N, 1) or 1D arrays (N,). Accept numpy arrays or torch tensors.
    title : str
        Plot title.
    identity_line : bool
        Draw the y = x identity line.
    fit_line : bool
        Draw an OLS fit line y = a*x + b and report (a, b) in the return dict.
    equal_limits : bool
        Apply equal x/y limits for visual comparability.
    sample : int or None
        If smaller than N, randomly subsample that many points for plotting (metrics are computed on full data).
    random_state : int or None
        RNG seed for subsampling.
    s, alpha : float
        Marker size and transparency for the scatter.
    show_metrics_on_figure : bool
        If True, annotate selected metrics inside the plot. If False, produce a clean figure.
    metrics : dict or None
        Optional precomputed metrics (e.g., from `compute_regression_metrics`). If None, they are computed here.
    metrics_to_show : sequence of str or None
        Which metrics to annotate. Defaults to ('mae','rmse','bias','r2','pearson_r','n').

    Returns
    -------
    Dict[str, float]
        The metrics dict (computed or the one provided), extended with 'slope' and 'intercept'.
    """
    def _to_numpy_1d(a: ArrayLike) -> np.ndarray:
        return np.ravel(to_numpy(a))

    y_true = _to_numpy_1d(y_true_col)
    y_pred = _to_numpy_1d(y_pred_col)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    n = y_true.size
    if n == 0:
        raise ValueError("No finite pairs remain after filtering.")

    # Compute metrics once (full data)
    if metrics is None:
        metrics = compute_regression_metrics(y_pred, y_true)
    if metrics_to_show is None:
        metrics_to_show = ("mae", "rmse", "bias", "r2", "pearson_r", "n")

    # Fit line (for visualization)
    slope = np.nan
    intercept = np.nan
    if fit_line and n >= 2:
        slope, intercept = np.polyfit(y_true, y_pred, deg=1)

    # Subsample for visualization only
    idx = np.arange(n)
    if sample is not None and 0 < sample < n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, size=sample, replace=False)

    xs = y_true[idx]
    ys = y_pred[idx]

    # Limits with margin
    xy_min = float(min(y_true.min(), y_pred.min()))
    xy_max = float(max(y_true.max(), y_pred.max()))
    span = xy_max - xy_min
    margin = 0.05 * (span if span > 0 else 1.0)
    lo, hi = xy_min - margin, xy_max + margin

    # Scatter
    ax.scatter(xs, ys, s=s, alpha=alpha)

    # Identity line
    if identity_line:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, label="y = x")

    # Fit line
    if fit_line and np.isfinite(slope) and np.isfinite(intercept):
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                linestyle="-.", linewidth=1.0,
                label=f"fit: y = {slope:.3f}x + {intercept:.3f}")

    # Cosmetics
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

    if equal_limits:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc="best", frameon=True)

    # Annotate metrics (optional)
    if show_metrics_on_figure:
        # Build compact label with the chosen metrics
        kv = []
        for k in metrics_to_show:
            if k not in metrics:
                continue
            val = metrics[k]
            if k in ("n",):
                kv.append(f"{k}={int(val)}")
            elif k in ("r2", "pearson_r"):
                kv.append(f"{k}={val:.4f}")
            else:
                kv.append(f"{k}={val:.4g}")
        text = "\n".join(kv)
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.5", alpha=0.85),
        )

    # Return metrics extended with fit params
    out = dict(metrics)
    out["slope"] = float(slope)
    out["intercept"] = float(intercept)
    out["x_min"] = float(lo)
    out["x_max"] = float(hi)
    return out




def plot_train_test_loss(
    loss_dict: Mapping[str, Union[Sequence[Any], np.ndarray, torch.Tensor, Mapping[str, Union[Sequence[Any], np.ndarray, torch.Tensor]]]],
    *,
    title: str = "Training/Validation Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    yscale: str = "log",             # "log" or "linear"
    smoothing: Optional[str] = None, # None | "ema" | "moving_avg"
    ema_alpha: float = 0.2,          # for EMA smoothing
    moving_avg_window: int = 5,      # for moving average smoothing
    mark_min: bool = True,           # mark the epoch of minimal loss for each series
    grid: bool = True,
    legend_loc: str = "best",
    # Saving / rendering
    save: bool = False,              # backward-compat; if True and save_path is None -> "./plots/train_test_losses.png"
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    # Reuse axes if desired
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Dict[str, float]]]:
    """
    Plot one or multiple loss curves.

    Parameters
    ----------
    loss_dict : mapping
        Dictionary of loss series. Two accepted shapes:
        1) {"Model A": losses, "Model B": losses}
        2) {"Model A": {"train": train_losses, "val": val_losses}, ...}
       Each "losses" can be a list/tuple, numpy array, torch tensor, or a list of tensors.
    title, xlabel, ylabel : str
        Plot labels.
    yscale : {"log", "linear"}
        Y axis scale.
    smoothing : {None, "ema", "moving_avg"}
        Optional smoothing to apply for visualization (does not change the raw data).
    ema_alpha : float
        EMA smoothing factor in (0,1]. Larger = more smoothing.
    moving_avg_window : int
        Window size for moving average; ignored unless smoothing == "moving_avg".
    mark_min : bool
        If True, annotate and mark the epoch of the minimum loss per series.
    grid : bool
        Show a light grid.
    legend_loc : str
        Legend location.
    save : bool
        Backward-compatible toggle. If True and `save_path` is None, defaults to "./plots/train_test_losses.png".
    save_path : Optional[str]
        If provided, save the figure to this path (directories created as needed).
    dpi : int
        Figure DPI when saving.
    show : bool
        If True, call plt.show() at the end.
    ax : Optional[matplotlib.axes.Axes]
        Existing axes to draw on; if None, a new figure/axes is created.

    Returns
    -------
    fig, ax, summary : (Figure, Axes, dict)
        - fig / ax: the Matplotlib figure and axes used.
        - summary: per-series information, e.g. {"Model A": {"min": 0.01, "argmin": 42, "last": 0.012}, ...}

    Notes
    -----
    - Smoothing is only for visualization; reported summary values come from the *raw* series.
    - When `save=True` and `save_path is None`, the file is saved to "./plots/train_test_losses.png".
    """

    def _to_1d_float_array(x: Union[Sequence[Any], np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert various inputs to a 1D float numpy array."""
        arr = to_numpy(x)
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            arr = np.asarray([float(t.detach().cpu().item()) for t in x], dtype=float)
        return np.asarray(arr, dtype=float).reshape(-1)

    def _smooth(y: np.ndarray) -> np.ndarray:
        """Apply optional smoothing to a 1D array for plotting only."""
        if smoothing is None:
            return y
        if smoothing == "ema":
            if not (0.0 < ema_alpha <= 1.0):
                raise ValueError("ema_alpha must be in (0, 1].")
            out = np.empty_like(y)
            out[0] = y[0]
            for i in range(1, len(y)):
                out[i] = ema_alpha * y[i] + (1.0 - ema_alpha) * out[i - 1]
            return out
        if smoothing == "moving_avg":
            w = int(moving_avg_window)
            if w <= 1:
                return y
            kernel = np.ones(w, dtype=float) / float(w)
            # 'same' to keep the same length; edges are averaged with fewer points
            return np.convolve(y, kernel, mode="same")
        raise ValueError(f"Unknown smoothing mode: {smoothing}")

    # Prepare figure/axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True
    else:
        fig = ax.figure

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)

    summary: Dict[str, Dict[str, float]] = {}

    # Normalize loss_dict to flat series for plotting
    # Case 1: value is a sequence/array/tensor => single series
    # Case 2: value is a mapping => multiple named series (e.g., train/val/test)
    for model_name, series in loss_dict.items():
        if isinstance(series, Mapping):
            # nested: plot each split
            for split_name, split_series in series.items():
                y_raw = _to_1d_float_array(split_series)
                epochs = np.arange(1, len(y_raw) + 1)
                y_vis = _smooth(y_raw)

                ax.plot(epochs, y_vis, label=f"{model_name} ({split_name})")

                min_idx = int(np.argmin(y_raw))
                min_val = float(y_raw[min_idx])
                if mark_min:
                    ax.scatter(epochs[min_idx], y_vis[min_idx], marker="o", s=25, zorder=3)
                    ax.annotate(
                        f"min={min_val:.3e} @ {epochs[min_idx]}",
                        xy=(epochs[min_idx], y_vis[min_idx]),
                        xytext=(5, 8),
                        textcoords="offset points",
                        fontsize=8,
                    )

                # record summary per series
                summary[f"{model_name} ({split_name})"] = {
                    "min": min_val,
                    "argmin": float(min_idx + 1),  # epoch index (1-based)
                    "last": float(y_raw[-1]),
                    "n_epochs": float(len(y_raw)),
                }
        else:
            # flat: single series per model_name
            y_raw = _to_1d_float_array(series)
            epochs = np.arange(1, len(y_raw) + 1)
            y_vis = _smooth(y_raw)

            ax.plot(epochs, y_vis, label=f"{model_name}")

            min_idx = int(np.argmin(y_raw))
            min_val = float(y_raw[min_idx])
            if mark_min:
                ax.scatter(epochs[min_idx], y_vis[min_idx], marker="o", s=25, zorder=3)
                ax.annotate(
                    f"min={min_val:.3g} @ {epochs[min_idx]}",
                    xy=(epochs[min_idx], y_vis[min_idx]),
                    xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=8,
                )

            summary[model_name] = {
                "min": min_val,
                "argmin": float(min_idx + 1),
                "last": float(y_raw[-1]),
                "n_epochs": float(len(y_raw)),
            }

    if grid:
        ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc=legend_loc, frameon=True)
    fig.tight_layout()

    # Saving logic
    final_save_path = save_path
    if save and final_save_path is None:
        final_save_path = os.path.join(".", "plots", "train_test_losses.png")
    if final_save_path is not None:
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        fig.savefig(final_save_path, dpi=dpi)

    if show:
        plt.show()
    elif created_fig:
        # If we created the figure and user does not want to show it,
        # it's still returned so the caller can manage it.
        pass

    return fig, ax, summary
