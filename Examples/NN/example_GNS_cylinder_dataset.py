#!/usr/bin/env python
"""
Example of GNS for a cylinder dataset.

Parallel to example_MLP_xfoil_dataset.py / example_KAN_xfoil_dataset.py,
using cylinder train/test graph-ready files and the config-driven GNS interface.

Preprocessing assumptions
-------------------------
1) Dataset split is expected to be done beforehand with:
   `Converters/tsuite_train_test_split.py`
   producing `CYLINDER_TRAIN.h5` and `CYLINDER_TEST.h5`.

2) Graph handling in this example:
   - The graph is read from group `/GRAPH` inside `CYLINDER_TRAIN.h5`.
   - If `/GRAPH` does not exist there, the script creates the graph online
     from `CYLINDER.h5` using `Graph.from_pyLOM_mesh(...)`, stores it inside
     `CYLINDER_TRAIN.h5`, then continues.
"""

from dataclasses import asdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from dacite import Config as DaciteConfig, from_dict

import pyLOM
from pyLOM import Mesh
from pyLOM.NN import (
    Dataset,
    GNS,
    Graph,
    MinMaxScaler,
    Pipeline,
    RegressionEvaluator,
    RobustScaler,
    StandardScaler,
)
from pyLOM.NN.utils.config_schema import GNSModelConfig, GNSTrainingConfig
from pyLOM.NN.utils.experiment import plot_train_test_loss, plot_true_vs_pred, save_experiment_artifacts
from pyLOM.utils import raiseError
from pyLOM.utils.config_resolvers import load_yaml

# Load configuration
cfg_path = (Path(__file__).resolve().parent / "configs" / "example_GNS_cylinder_dataset_config.yaml").resolve()
cfg = load_yaml(cfg_path)

exp_cfg = cfg["experiment"]
dset_cfg = cfg["datasets"]
data_cfg = cfg["dataset_config"]
model_section = cfg["model"]
train_section = cfg["training"]

# Resolve input/output paths
train_ds_path = Path(dset_cfg["train_ds"])
if not train_ds_path.is_absolute():
    train_ds_path = (cfg_path.parent / train_ds_path).resolve()
test_ds_path = Path(dset_cfg["test_ds"])
if not test_ds_path.is_absolute():
    test_ds_path = (cfg_path.parent / test_ds_path).resolve()
val_ds_raw = dset_cfg.get("val_ds")
if val_ds_raw in (None, ""):
    val_ds_path = None
else:
    val_ds_path = Path(val_ds_raw)
    if not val_ds_path.is_absolute():
        val_ds_path = (cfg_path.parent / val_ds_path).resolve()

# For this example, graph provenance lives inside TRAIN dataset (/GRAPH group).
graph_path = train_ds_path
results_path = Path(exp_cfg["results_path"])
if not results_path.is_absolute():
    results_path = (cfg_path.parent / results_path).resolve()

required_dataset_paths = [train_ds_path, test_ds_path]
if val_ds_path is not None:
    required_dataset_paths.append(val_ds_path)
for p in required_dataset_paths:
    if not p.exists():
        raiseError(
            f"Required dataset file not found: {p}\n"
            "For cylinder examples, generate split files first with:\n"
            "  Converters/tsuite_train_test_split.py"
        )

# Build graph online (didactic path), overwrite /GRAPH in TRAIN file, then read it from HDF5.
source_dataset = train_ds_path.with_name("CYLINDER.h5")
if not source_dataset.exists():
    raiseError(
        f"Automatic graph creation requires base dataset: {source_dataset}\n"
        "Create/keep CYLINDER.h5 and re-run this example."
    )

pyLOM.pprint(0, f"overwriting graph group in {train_ds_path}")
pyLOM.pprint(0, f"Building graph from mesh in: {source_dataset}")
mesh = Mesh.load(str(source_dataset), mpio=False)
graph = Graph.from_pyLOM_mesh(mesh=mesh, device="cpu")
with h5py.File(str(train_ds_path), "a") as f:
    if "GRAPH" in f:
        del f["GRAPH"]
graph.save(str(train_ds_path), mode="a")
with h5py.File(str(train_ds_path), "r") as f:
    if "GRAPH" not in f:
        raiseError(f"Failed to store /GRAPH in {train_ds_path}")
pyLOM.pprint(0, f"Graph stored in /GRAPH inside: {train_ds_path}")

# Build datasets
if data_cfg.get("mesh_shape") is not None:
    mesh_shape = tuple(data_cfg["mesh_shape"])
else:
    mesh_for_shape = Mesh.load(str(train_ds_path), mpio=False)
    mesh_shape = (int(mesh_for_shape.ncells),)

inputs_scaler = None
if bool(data_cfg.get("scale_inputs", False)):
    input_scaler_type = str(data_cfg.get("input_scaler_type", "none")).strip().lower()
    if input_scaler_type in {"none", ""}:
        inputs_scaler = None
    elif input_scaler_type == "minmax":
        inputs_scaler = MinMaxScaler()
    elif input_scaler_type == "standard":
        inputs_scaler = StandardScaler()
    elif input_scaler_type == "robust":
        inputs_scaler = RobustScaler()
    else:
        raiseError(f"Unsupported input_scaler_type='{input_scaler_type}'. Use: none|minmax|standard|robust")

outputs_scaler = None
if bool(data_cfg.get("scale_outputs", False)):
    output_scaler_type = str(data_cfg.get("output_scaler_type", "none")).strip().lower()
    if output_scaler_type in {"none", ""}:
        outputs_scaler = None
    elif output_scaler_type == "minmax":
        outputs_scaler = MinMaxScaler()
    elif output_scaler_type == "standard":
        outputs_scaler = StandardScaler()
    elif output_scaler_type == "robust":
        outputs_scaler = RobustScaler()
    else:
        raiseError(f"Unsupported output_scaler_type='{output_scaler_type}'. Use: none|minmax|standard|robust")

ds_kwargs = {
    "field_names": data_cfg["field_names"],
    "variables_names": data_cfg.get("variables_names", ["all"]),
    "add_variables": bool(data_cfg.get("add_variables", True)),
    "add_mesh_coordinates": bool(data_cfg.get("add_mesh_coordinates", False)),
    "mesh_shape": mesh_shape,
    "inputs_scaler": inputs_scaler,
    "outputs_scaler": outputs_scaler,
    "squeeze_last_dim": False,
    "channels_last": True,
}

td_train = Dataset.load(str(train_ds_path), **ds_kwargs)
td_test = Dataset.load(str(test_ds_path), **ds_kwargs)
td_val = Dataset.load(str(val_ds_path), **ds_kwargs) if val_ds_path is not None else None

# Build model and training config
dcfg = DaciteConfig(strict=True)
model_cfg = from_dict(GNSModelConfig, model_section["config"], config=dcfg)
train_cfg = from_dict(GNSTrainingConfig, train_section, config=dcfg)

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
    extra_files={},
    return_path=True,
)

out_dir = Path(out_dir)
fig, _, _ = plot_train_test_loss(
    {"GNS": {"train": logs.get("train_loss", []), "val": logs.get("test_loss", [])}},
    save=True,
    save_path=str(out_dir / "train_test_loss.png"),
    show=False,
    yscale="log",
)
plt.close(fig)
plot_true_vs_pred(y_true_flat, y_pred_flat, str(out_dir / "true_vs_pred.png"))

pyLOM.pprint(0, f"Artifacts saved in: {out_dir}")
pyLOM.cr_info()
