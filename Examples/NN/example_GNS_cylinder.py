#!/usr/bin/env python
#
# Example of GNS for a cylinder dataset.
# 
# Parallel to example_MLP_xfoil_dataset.py / example_KAN_xfoil_dataset.py,
# using cylinder train/test graph-ready files and the config-driven GNS interface.
# 
# Preprocessing assumptions
# -------------------------
# 1) Dataset split is expected to be done beforehand with:
#    `Converters/tsuite_train_test_split.py`
#    producing `CYLINDER_TRAIN.h5` and `CYLINDER_TEST.h5`.
# 
# 2) Graph handling in this example:
#    - The graph is read from group `/GRAPH` inside `CYLINDER_TRAIN.h5`.
#    - If `/GRAPH` does not exist there, the script creates the graph online
#      from `CYLINDER.h5` using `Graph.from_pyLOM_mesh(...)`, stores it inside
#      `CYLINDER_TRAIN.h5`, then continues.
#
# Last revision: 23/04/2026
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import asdict
from pathlib     import Path

import pyLOM, pyLOM.NN, pyLOM.utils

device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


# Load configuration
cfg_path = '.configs/example_GNS_cylinder_config.yaml'
cfg      = pyLOM.NN.load_yaml(cfg_path)

exp_cfg       = cfg["experiment"]
dset_cfg      = cfg["datasets"]
data_cfg      = cfg["dataset_config"]
model_section = cfg["model"]
train_section = cfg["training"]

# Resolve input/output paths
train_ds_path = dset_cfg["train_ds"]
test_ds_path  = dset_cfg["test_ds"]
val_ds_path   = dset_cfg["val_ds"]

# For this example, graph provenance lives inside TRAIN dataset (/GRAPH group).
graph_path   = train_ds_path
results_path = exp_cfg["results_path"]
required_dataset_paths = [train_ds_path, test_ds_path]
if val_ds_path is not None:
    required_dataset_paths.append(val_ds_path)

# Build graph online (didactic path), overwrite /GRAPH in TRAIN file, then read it from HDF5.
source_dataset = train_ds_path.with_name("CYLINDER.h5")
pyLOM.pprint(0, f"Building graph from mesh in: {source_dataset}")
mesh  = pyLOM.Mesh.load(str(source_dataset), mpio=False)
graph = pyLOM.NN.Graph.from_pyLOM_mesh(mesh=mesh,device=device)
graph.save(str(train_ds_path), mode="a")
pyLOM.pprint(0, f"Graph stored in /GRAPH inside: {train_ds_path}")

# Build datasets
mesh_shape = tuple(data_cfg["mesh_shape"]) if data_cfg.get("mesh_shape") is not None else (int(mesh.ncellsG),)

inputs_scaler = None
if bool(data_cfg.get("scale_inputs", False)):
    input_scaler_type = str(data_cfg.get("input_scaler_type", "none")).strip().lower()
    if input_scaler_type in {"none", ""}:
        inputs_scaler = None
    elif input_scaler_type == "minmax":
        inputs_scaler = pyLOM.NN.MinMaxScaler()
    elif input_scaler_type == "standard":
        inputs_scaler = pyLOM.NN.StandardScaler()
    elif input_scaler_type == "robust":
        inputs_scaler = pyLOM.NN.RobustScaler()
    else:
        pyLOM.utils.raiseError(f"Unsupported input_scaler_type='{input_scaler_type}'. Use: none|minmax|standard|robust")

outputs_scaler = None
if bool(data_cfg.get("scale_outputs", False)):
    output_scaler_type = str(data_cfg.get("output_scaler_type", "none")).strip().lower()
    if output_scaler_type in {"none", ""}:
        outputs_scaler = None
    elif output_scaler_type == "minmax":
        outputs_scaler = pyLOM.NN.MinMaxScaler()
    elif output_scaler_type == "standard":
        outputs_scaler = pyLOM.NN.StandardScaler()
    elif output_scaler_type == "robust":
        outputs_scaler = pyLOM.NN.RobustScaler()
    else:
        pyLOM.utils.raiseError(f"Unsupported output_scaler_type='{output_scaler_type}'. Use: none|minmax|standard|robust")

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
    extra_files={},
    return_path=True,
)

out_dir = Path(out_dir)
fig, _, _ = pyLOM.NN.plot_train_test_loss(
    {"GNS": {"train": logs.get("train_loss", []), "val": logs.get("test_loss", [])}},
    save=True,
    save_path=str(out_dir / "train_test_loss.png"),
    show=False,
    yscale="log",
)
plt.close(fig)
pyLOM.NN.plot_true_vs_pred(y_true_flat, y_pred_flat, str(out_dir / "true_vs_pred.png"))

pyLOM.pprint(0, f"Artifacts saved in: {out_dir}")
pyLOM.cr_info()
