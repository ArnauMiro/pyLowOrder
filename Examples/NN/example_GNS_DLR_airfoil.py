#!/usr/bin/env python
"""
Example of GNS for DLR/NLR-style airfoil datasets.

Parallel to example_MLP_DLR_airfoil.py and example_KAN_DLR_airfoil.py,
but using the config-driven GNS interface.

Dataset prerequisite
--------------------
This example expects the DLR/NLR HDF5 files already preprocessed with:
`Converters/DLR2h5.py`

Expected files:
  - ../../../DATA/TRAIN_converter.h5
  - ../../../DATA/TEST_converter.h5
  - ../../../DATA/VAL_converter.h5

The converter stores a `GRAPH` group in each file. This script reads the graph
from `TRAIN_converter.h5`.
"""

from dataclasses import asdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import pyLOM
from pyLOM.NN import Dataset, GNS, MinMaxScaler, Pipeline, RegressionEvaluator, RobustScaler, StandardScaler
from pyLOM.NN.utils.config_schema import GNSModelConfig, GNSTrainingConfig
from pyLOM.NN.utils.experiment import plot_train_test_loss, plot_true_vs_pred, save_experiment_artifacts
from pyLOM.utils import raiseError
from pyLOM.utils.config_resolvers import load_yaml, dataclass_from_dict


def _resolve(path_like: str, cfg_path: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (cfg_path.parent / p).resolve()


def _build_scaler(enabled: bool, scaler_type: str):
    if not enabled:
        return None
    st = str(scaler_type).strip().lower()
    if st in {"none", ""}:
        return None
    if st == "minmax":
        return MinMaxScaler()
    if st == "standard":
        return StandardScaler()
    if st == "robust":
        return RobustScaler()
    raiseError(f"Unsupported scaler_type='{scaler_type}'. Use: none|minmax|standard|robust")


def _infer_mesh_shape_from_h5(train_ds_path: Path) -> tuple[int]:
    with h5py.File(str(train_ds_path), "r") as f:
        if "/DATASET/npoints" in f:
            npoints = int(f["/DATASET/npoints"][0])
        else:
            npoints = int(f["/DATASET/xyz"].shape[0])
    return (npoints,)


def _load_converter_dataset(
    fname: Path,
    *,
    field_names: list[str],
    variables_names: list[str],
    mesh_shape: tuple[int],
    inputs_scaler,
    outputs_scaler,
) -> Dataset:
    with h5py.File(str(fname), "r") as f:
        fields_group = f["/DATASET/FIELDS"]
        vars_group = f["/DATASET/VARIABLES"]

        missing_fields = [name for name in field_names if name not in fields_group]
        if missing_fields:
            raiseError(f"Missing field(s) {missing_fields} in {fname}")

        missing_vars = [name for name in variables_names if name not in vars_group]
        if missing_vars:
            raiseError(f"Missing variable(s) {missing_vars} in {fname}")

        variables_out = tuple(np.asarray(fields_group[name]["value"][:], dtype=np.float32) for name in field_names)
        variables_in = None
        if len(variables_names) > 0:
            variables_in = np.stack(
                [np.asarray(vars_group[name]["value"][:], dtype=np.float32) for name in variables_names],
                axis=1,
            )

    return Dataset(
        variables_out=variables_out,
        variables_in=variables_in,
        mesh_shape=mesh_shape,
        inputs_scaler=inputs_scaler,
        outputs_scaler=outputs_scaler,
        snapshots_by_column=True,
        squeeze_last_dim=False,
        channels_last=True,
    )


# Load configuration
cfg_path = (Path(__file__).resolve().parent / "configs" / "example_GNS_DLR_airfoil_config.yaml").resolve()
cfg = load_yaml(cfg_path)

exp_cfg = cfg["experiment"]
dset_cfg = cfg["datasets"]
data_cfg = cfg["dataset_config"]
model_section = cfg["model"]
train_section = cfg["training"]

# Resolve input/output paths
train_ds_path = _resolve(dset_cfg["train_ds"], cfg_path)
test_ds_path = _resolve(dset_cfg["test_ds"], cfg_path)
val_ds_raw = dset_cfg.get("val_ds")
val_ds_path = None if val_ds_raw in (None, "") else _resolve(val_ds_raw, cfg_path)

graph_path = _resolve(model_section["graph_path"], cfg_path)
results_path = _resolve(exp_cfg["results_path"], cfg_path)

required_paths = [train_ds_path, test_ds_path, graph_path]
if val_ds_path is not None:
    required_paths.append(val_ds_path)
for p in required_paths:
    if not p.exists():
        raiseError(f"Required input file not found: {p}")

# Build datasets
mesh_shape = tuple(data_cfg["mesh_shape"]) if data_cfg.get("mesh_shape") is not None else _infer_mesh_shape_from_h5(train_ds_path)

inputs_scaler = _build_scaler(bool(data_cfg.get("scale_inputs", False)), data_cfg.get("input_scaler_type", "none"))
outputs_scaler = _build_scaler(bool(data_cfg.get("scale_outputs", False)), data_cfg.get("output_scaler_type", "none"))

field_names = list(data_cfg["field_names"])
variables_names = list(data_cfg.get("variables_names", []))

td_train = _load_converter_dataset(
    train_ds_path,
    field_names=field_names,
    variables_names=variables_names,
    mesh_shape=mesh_shape,
    inputs_scaler=inputs_scaler,
    outputs_scaler=outputs_scaler,
)
td_test = _load_converter_dataset(
    test_ds_path,
    field_names=field_names,
    variables_names=variables_names,
    mesh_shape=mesh_shape,
    inputs_scaler=inputs_scaler,
    outputs_scaler=outputs_scaler,
)
td_val = (
    _load_converter_dataset(
        val_ds_path,
        field_names=field_names,
        variables_names=variables_names,
        mesh_shape=mesh_shape,
        inputs_scaler=inputs_scaler,
        outputs_scaler=outputs_scaler,
    )
    if val_ds_path is not None
    else None
)

# Build model and training config
model_cfg = dataclass_from_dict(GNSModelConfig, model_section["config"], strict=True)
train_cfg = dataclass_from_dict(GNSTrainingConfig, train_section, strict=True)

model = GNS.from_graph_path(config=model_cfg, graph_path=graph_path)

pipeline = Pipeline(
    train_dataset=td_train,
    valid_dataset=td_val if td_val is not None else td_test,
    test_dataset=td_test,
    model=model,
    training_params={"config": train_cfg},
)
logs = pipeline.run()

# Evaluate
y_pred = pipeline.model.predict(td_test)
y_true = td_test[:][1]
y_true = td_test.inverse_scale_outputs(y_true)
y_pred = td_test.inverse_scale_outputs(y_pred)

y_true_np = np.asarray(y_true, dtype=np.float64)
y_pred_np = np.asarray(y_pred, dtype=np.float64)
y_true_flat = y_true_np.reshape(-1, y_true_np.shape[-1])
y_pred_flat = y_pred_np.reshape(-1, y_pred_np.shape[-1])

evaluator = RegressionEvaluator()
metrics = evaluator(y_true_flat, y_pred_flat)
evaluator.print_metrics()


def _save_loss_plot(path: Path) -> None:
    fig, _, _ = plot_train_test_loss(
        {"GNS": {"train": logs.get("train_loss", []), "val": logs.get("test_loss", [])}},
        save=True,
        save_path=str(path),
        show=False,
        yscale="log",
    )
    plt.close(fig)


# Save results
out_dir = save_experiment_artifacts(
    base_path=results_path,
    model=pipeline.model,
    metrics_dict={k: float(v) for k, v in metrics.items()},
    inputs_scaler=inputs_scaler,
    outputs_scaler=outputs_scaler,
    full_run_config={
        "experiment": exp_cfg,
        "datasets": dset_cfg,
        "dataset_config": data_cfg,
        "model": {"graph_path": str(graph_path), "config": asdict(model_cfg)},
        "training": asdict(train_cfg),
    },
    extra_files={
        "train_test_loss.png": _save_loss_plot,
        "true_vs_pred.png": lambda p: plot_true_vs_pred(y_true_flat, y_pred_flat, p),
    },
    return_path=True,
)

pyLOM.pprint(0, f"Artifacts saved in: {out_dir}")
pyLOM.cr_info()
