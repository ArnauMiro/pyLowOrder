#!/usr/bin/env python
"""
Example of GNS for a cylinder dataset.

Parallel to example_MLP_xfoil_dataset.py / example_KAN_xfoil_dataset.py,
using cylinder train/test graph-ready files and the config-driven GNS interface.

Preprocessing assumptions
-------------------------
1) Dataset split is expected to be done beforehand with:
   `Converters/split_cylinder_train_test.py`
   producing `CYLINDER_TRAIN.h5` and `CYLINDER_TEST.h5`.

2) Graph handling in this example:
   - The graph is read from group `/GRAPH` inside `CYLINDER_TRAIN.h5`.
   - If `/GRAPH` does not exist there, the script creates the graph online
     from `CYLINDER.h5` using `Graph.from_pyLOM_mesh(...)`, stores it inside
     `CYLINDER_TRAIN.h5`, then continues.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import h5py
from dacite import Config as DaciteConfig, from_dict

import pyLOM
from pyLOM import Mesh
from pyLOM.NN import (
    Dataset,
    GNS,
    Graph,
    Pipeline,
    RegressionEvaluator,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)
from pyLOM.NN.utils.config_schema import GNSModelConfig, GNSTrainingConfig
from pyLOM.NN.utils.experiment import (
    plot_train_test_loss,
    plot_true_vs_pred,
    save_experiment_artifacts,
)
from pyLOM.utils import pprint, raiseError
from pyLOM.utils.config_resolvers import load_yaml


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


def _infer_mesh_shape(train_ds_path: Path) -> tuple[int]:
    mesh = Mesh.load(str(train_ds_path), mpio=False)
    return (int(mesh.ncells),)


def _ensure_dataset_splits(train_ds_path: Path, test_ds_path: Path, val_ds_path: Path | None) -> None:
    required = [train_ds_path, test_ds_path]
    if val_ds_path is not None:
        required.append(val_ds_path)
    for p in required:
        if not p.exists():
            raiseError(
                f"Required dataset file not found: {p}\n"
                "For cylinder examples, generate split files first with:\n"
                "  Converters/split_cylinder_train_test.py"
            )


def _ensure_graph_in_train_dataset(train_ds_path: Path) -> None:
    # Do not call Graph.load() here: missing /GRAPH triggers MPI_ABORT in pyLOM.
    with h5py.File(str(train_ds_path), "r") as f:
        if "GRAPH" in f:
            return

    source_dataset = train_ds_path.with_name("CYLINDER.h5")
    if not source_dataset.exists():
        raiseError(
            f"Graph group /GRAPH not found in: {train_ds_path}\n"
            f"Automatic graph creation requires base dataset: {source_dataset}\n"
            "Create/keep CYLINDER.h5 and re-run this example."
        )

    pprint(0, f"Graph group /GRAPH not found. Building graph from mesh in: {source_dataset}")
    mesh = Mesh.load(str(source_dataset), mpio=False)
    graph = Graph.from_pyLOM_mesh(mesh=mesh, device="cpu")
    # Append /GRAPH to TRAIN file, preserving existing root groups.
    graph.save(str(train_ds_path), mode="a")
    with h5py.File(str(train_ds_path), "r") as f:
        if "GRAPH" not in f:
            raiseError(f"Failed to store /GRAPH in {train_ds_path}")
    pprint(0, f"Graph created and stored in /GRAPH inside: {train_ds_path}")


def main() -> None:
    cfg_path = (Path(__file__).resolve().parent / "configs" / "example_GNS_cylinder_dataset_config.yaml").resolve()
    cfg = load_yaml(cfg_path)

    exp_cfg = cfg["experiment"]
    dset_cfg = cfg["datasets"]
    data_cfg = cfg["dataset_config"]
    model_section = cfg["model"]
    train_section = cfg["training"]

    train_ds_path = _resolve(dset_cfg["train_ds"], cfg_path)
    test_ds_path = _resolve(dset_cfg["test_ds"], cfg_path)
    val_ds_raw = dset_cfg.get("val_ds")
    val_ds_path = None if val_ds_raw in (None, "") else _resolve(val_ds_raw, cfg_path)

    # For this example, graph provenance lives inside TRAIN dataset (/GRAPH group).
    graph_path = train_ds_path
    results_path = _resolve(exp_cfg["results_path"], cfg_path)

    _ensure_dataset_splits(train_ds_path, test_ds_path, val_ds_path)
    _ensure_graph_in_train_dataset(train_ds_path)

    mesh_shape = tuple(data_cfg["mesh_shape"]) if data_cfg.get("mesh_shape") is not None else _infer_mesh_shape(train_ds_path)

    inputs_scaler = _build_scaler(bool(data_cfg.get("scale_inputs", False)), data_cfg.get("input_scaler_type", "none"))
    outputs_scaler = _build_scaler(bool(data_cfg.get("scale_outputs", False)), data_cfg.get("output_scaler_type", "none"))

    ds_kwargs = {
        "field_names": data_cfg["field_names"],
        "variables_names": data_cfg.get("variables_names", ["all"]),
        "add_variables": bool(data_cfg.get("add_variables", True)),
        "add_mesh_coordinates": bool(data_cfg.get("add_mesh_coordinates", False)),
        "mesh_shape": mesh_shape,
        "inputs_scaler": inputs_scaler,
        "outputs_scaler": outputs_scaler,
        "squeeze_last_dim": False,
    }

    td_train = Dataset.load(str(train_ds_path), **ds_kwargs)
    td_test = Dataset.load(str(test_ds_path), **ds_kwargs)
    td_val = Dataset.load(str(val_ds_path), **ds_kwargs) if val_ds_path is not None else None

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

    pprint(0, f"Artifacts saved in: {out_dir}")
    pyLOM.cr_info()


if __name__ == "__main__":
    main()
