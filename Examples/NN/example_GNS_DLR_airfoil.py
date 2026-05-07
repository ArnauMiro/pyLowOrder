#!/usr/bin/env python
#
# Example of GNS for DLR/NLR-style airfoil datasets.
# 
# Parallel to example_MLP_DLR_airfoil.py and example_KAN_DLR_airfoil.py,
# but using the config-driven GNS interface.
# 
# Dataset prerequisite
# --------------------
# This example expects the DLR/NLR HDF5 files already preprocessed with:
# `Converters/DLR2h5.py`
# 
# Expected files:
#   - ./DATA/TRAIN_converter.h5
#   - ./DATA/TEST_converter.h5
#   - ./DATA/VAL_converter.h5
# 
# The converter stores a `GRAPH` group in each file. This script reads the graph
# from `TRAIN_converter.h5`.
#
# Last revision: 30/04/2026
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import asdict
from pathlib import Path

import pyLOM, pyLOM.NN

device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


def _build_scaler(enabled: bool, scaler_type: str):
    if not enabled:
        return None
    st = str(scaler_type).strip().lower()
    if st in {"none", ""}:
        return None
    if st == "minmax":
        return pyLOM.NN.MinMaxScaler()
    if st == "standard":
        return pyLOM.NN.StandardScaler()
    if st == "robust":
        return pyLOM.NN.RobustScaler()
    pyLOM.raiseError(f"Unsupported scaler_type='{scaler_type}'. Use: none|minmax|standard|robust")

def _save_loss_plot(path: Path) -> None:
    fig, _, _ = pyLOM.NN.plot_train_test_loss(
        {"GNS": {"train": logs.get("train_loss", []), "val": logs.get("test_loss", [])}},
        save=True,
        save_path=str(path),
        show=False,
        yscale="log",
    )
    plt.close(fig)


# Load configuration
cfg_path = '.configs/example_GNS_DLR_airfoil_config.yaml'
cfg      = pyLOM.NN.load_yaml(cfg_path)

exp_cfg       = cfg["experiment"]
dset_cfg      = cfg["datasets"]
data_cfg      = cfg["dataset_config"]
model_section = cfg["model"]
train_section = cfg["training"]

# Resolve input/output paths
train_ds_path = dset_cfg["train_ds"]
test_ds_path  = dset_cfg["test_ds"]
val_ds_path   = dset_cfg.get("val_ds")

graph_path   = train_ds_path
results_path = exp_cfg["results_path"]
required_dataset_paths = [train_ds_path, test_ds_path, graph_path]
if val_ds_path is not None:
    required_dataset_paths.append(val_ds_path)

# Build datasets
d = pyLOM.Dataset.load(train_ds_path,mpio=False)
mesh_shape = d.xyz.shape[0]

inputs_scaler  = _build_scaler(bool(data_cfg.get("scale_inputs", False)),  data_cfg.get("input_scaler_type", "none"))
outputs_scaler = _build_scaler(bool(data_cfg.get("scale_outputs", False)), data_cfg.get("output_scaler_type", "none"))

ds_kwargs = {
    "field_names": list(data_cfg["field_names"]),
    "variables_names": list(data_cfg.get("variables_names", [])),
    "mesh_shape": mesh_shape,
    "inputs_scaler": inputs_scaler,
    "outputs_scaler": outputs_scaler,
}

td_train = pyLOM.Dataset.load(str(train_ds_path), **ds_kwargs)
td_test  = pyLOM.Dataset.load(str(test_ds_path), **ds_kwargs)
td_val   = pyLOM.Dataset.load(str(val_ds_path), **ds_kwargs) if val_ds_path is not None else None

# Build model and training config
model_cfg = pyLOM.NN.dataclass_from_dict(pyLOM.NN.GNSModelConfig, model_section["config"], strict=True)
train_cfg = pyLOM.NN.dataclass_from_dict(pyLOM.NN.GNSTrainingConfig, train_section, strict=True)

model = pyLOM.NN.GNS.from_graph_path(config=model_cfg, graph_path=graph_path)

pipeline = pyLOM.NN.Pipeline(
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

evaluator = pyLOM.NN.RegressionEvaluator()
metrics = evaluator(y_true_flat, y_pred_flat)
evaluator.print_metrics()

# Save results
out_dir = pyLOM.NN.save_experiment_artifacts(
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
        "true_vs_pred.png": lambda p: pyLOM.NN.plot_true_vs_pred(y_true_flat, y_pred_flat, p),
    },
    return_path=True,
)

pyLOM.pprint(0, f"Artifacts saved in: {out_dir}")
pyLOM.cr_info()
