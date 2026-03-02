#!/usr/bin/env python
"""
Minimal end-to-end GNS example on CYLINDER train/test splits.

Flow:
1) Read Examples/NN/configs/example_GNS_config.yaml
2) Load datasets with pyLOM.NN.Dataset.load
3) Build GNS from graph_path and train it through Pipeline
4) Save artifacts with pyLOM.NN.utils.experiment helpers
5) Plot train/validation losses and a prediction heatmap
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dacite import Config as DaciteConfig, from_dict

from pyLOM.NN import Dataset, GNS, Pipeline, RegressionEvaluator
from pyLOM.NN.utils.config_schema import GNSModelConfig, GNSTrainingConfig
from pyLOM.NN.utils.experiment import (
    plot_train_test_loss,
    save_experiment_artifacts,
    to_numpy,
)
from pyLOM.utils import pprint, raiseError
from pyLOM.utils.config_resolvers import load_yaml


def _plot_prediction_heatmap(
    xy: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> None:
    err = y_pred - y_true

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    cmap = "turbo"

    for ax, title, values in zip(
        axs,
        ("True", "Predicted", "Error"),
        (y_true, y_pred, err),
    ):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=6, cmap=cmap, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("GNS Prediction Map (first test snapshot)", fontsize=13)
    if save_path is not None:
        fig.savefig(save_path, dpi=250)
        plt.close(fig)


def main() -> None:
    cfg_path = (Path(__file__).resolve().parent / "configs" / "example_GNS_config.yaml").resolve()
    cfg = load_yaml(cfg_path)

    dcfg = DaciteConfig(strict=True)
    model_cfg = from_dict(GNSModelConfig, cfg["model"]["config"], config=dcfg)
    training_cfg = from_dict(GNSTrainingConfig, cfg["training"], config=dcfg)

    train_ds_path = (cfg_path.parent / cfg["datasets"]["train_ds"]).resolve()
    test_ds_path = (cfg_path.parent / cfg["datasets"]["test_ds"]).resolve()
    val_ds_raw = cfg["datasets"].get("val_ds")
    graph_path = (cfg_path.parent / cfg["model"]["graph_path"]).resolve()
    results_path = (cfg_path.parent / cfg["experiment"]["results_path"]).resolve()

    if val_ds_raw is None:
        val_ds_path = None
    else:
        val_ds_path = (cfg_path.parent / val_ds_raw).resolve()

    for path in (train_ds_path, test_ds_path, graph_path):
        if not path.exists():
            raiseError(f"Required input file not found: {path}")

    dataset_cfg = cfg["dataset_config"]
    mesh_shape = tuple(dataset_cfg["mesh_shape"]) if dataset_cfg.get("mesh_shape") else None

    ds_kwargs = {
        "field_names": dataset_cfg["field_names"],
        "variables_names": dataset_cfg.get("variables_names", ["all"]),
        "add_variables": bool(dataset_cfg.get("add_variables", True)),
        "add_mesh_coordinates": bool(dataset_cfg.get("add_mesh_coordinates", False)),
        "squeeze_last_dim": False,
    }
    if mesh_shape is not None:
        ds_kwargs["mesh_shape"] = mesh_shape

    ds_train = Dataset.load(str(train_ds_path), **ds_kwargs)
    ds_test = Dataset.load(str(test_ds_path), **ds_kwargs)
    ds_val = Dataset.load(str(val_ds_path), **ds_kwargs) if val_ds_path is not None else None

    model = GNS.from_graph_path(config=model_cfg, graph_path=graph_path)

    if int(model.graph.num_nodes) != int(ds_train.shape[1]):
        raiseError(
            f"Graph nodes ({int(model.graph.num_nodes)}) != dataset mesh nodes ({int(ds_train.shape[1])})."
        )

    pipeline = Pipeline(
        train_dataset=ds_train,
        valid_dataset=ds_val,
        test_dataset=ds_test,
        model=model,
        training_params={"config": training_cfg},
    )
    logs = pipeline.run()

    test_batch = ds_test[:]
    if isinstance(test_batch, tuple):
        _, y_true = test_batch
    else:
        y_true = test_batch

    y_pred = pipeline.model.predict(ds_test)
    y_true_np = np.asarray(to_numpy(ds_test.inverse_scale_outputs(y_true)), dtype=np.float64)
    y_pred_np = np.asarray(to_numpy(ds_test.inverse_scale_outputs(y_pred)), dtype=np.float64)

    y_true_flat = y_true_np.reshape(-1, y_true_np.shape[-1])
    y_pred_flat = y_pred_np.reshape(-1, y_pred_np.shape[-1])

    evaluator = RegressionEvaluator()
    metrics = evaluator(y_true_flat, y_pred_flat)
    evaluator.print_metrics()

    xy = model.graph.node_features_dict["xyz"].detach().cpu().numpy()
    if xy.shape[1] < 2:
        raiseError(f"Expected at least 2D coordinates for heatmap, got shape={xy.shape}.")
    xy2 = xy[:, :2]

    train_loss = logs.get("train_loss", [])
    val_loss = logs.get("test_loss", [])
    loss_series = {}
    if len(train_loss) > 0:
        loss_series["train"] = train_loss
    if len(val_loss) > 0:
        loss_series["val"] = val_loss
    loss_plot_dict = {"GNS": loss_series} if len(loss_series) > 0 else None

    out_dir = save_experiment_artifacts(
        base_path=results_path,
        model=pipeline.model,
        metrics_dict={k: float(v) for k, v in metrics.items()},
        full_run_config={
            "config_path": str(cfg_path),
            "experiment": cfg.get("experiment", {}),
            "datasets": cfg.get("datasets", {}),
            "dataset_config": dataset_cfg,
            "model": {
                "graph_path": str(graph_path),
                "config": asdict(model_cfg),
            },
            "training": asdict(training_cfg),
        },
        extra_files={
            **(
                {
                    "train_val_loss.png": lambda p: plot_train_test_loss(
                        loss_plot_dict,
                        save=True,
                        save_path=str(p),
                        show=False,
                        yscale="log",
                    )
                }
                if loss_plot_dict is not None
                else {}
            ),
            "prediction_heatmap.png": lambda p: _plot_prediction_heatmap(
                xy2,
                y_true_np[0, :, 0],
                y_pred_np[0, :, 0],
                Path(p),
            ),
        },
        return_path=True,
    )

    pprint(0, f"Artifacts saved in: {out_dir}")
    pprint(0, f"Model graph nodes: {int(model.graph.num_nodes)}, edges: {int(model.graph.num_edges)}")
    pprint(0, f"Train snapshots: {len(ds_train)} | Test snapshots: {len(ds_test)}")

    if cfg.get("experiment", {}).get("show_plots", True):
        if loss_plot_dict is not None:
            plot_train_test_loss(
                loss_plot_dict,
                show=True,
                save=False,
                yscale="log",
            )
        _plot_prediction_heatmap(xy2, y_true_np[0, :, 0], y_pred_np[0, :, 0], save_path=None)
        plt.show()


if __name__ == "__main__":
    main()
