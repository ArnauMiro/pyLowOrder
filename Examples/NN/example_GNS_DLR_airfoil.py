#!/usr/bin/env python

"""
Main script to train or optimize a GNS model using pyLOM v2.

Supports:
- Training from a YAML config file
- Hyperparameter optimization via Optuna
- Evaluation, saving metrics and artifacts
- Reloading the model and running inference

Author: Pablo Yeste
Last modified: 2025-07-16
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import optuna
import json
import pickle
from pathlib import Path
from typing import Union, Optional, Dict, Callable, Any

import pyLOM
from pyLOM.utils import cr_info
from pyLOM.NN import set_seed
from pyLOM.NN import GNS, Graph, MinMaxScaler, OptunaOptimizer, Pipeline
from pyLOM.NN.utils import RegressionEvaluator
from pyLOM.NN.gns import GNSConfig, TrainingConfig


def load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML file into a Python dictionary."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset(path: Union[str, Path], input_scaler: Any, output_scaler: Any) -> pyLOM.NN.Dataset:
    """
    Load a dataset and apply scaling.

    Args:
        path: Path to the dataset file.
        input_scaler: Scaler object for input features.
        output_scaler: Scaler object for output targets.

    Returns:
        A pyLOM-compatible dataset object.
    """
    return pyLOM.NN.Dataset.load(
        path,
        field_names=["CP"],
        add_variables=True,
        add_mesh_coordinates=False,
        variables_names=["AoA", "Mach"],
        inputs_scaler=input_scaler,
        outputs_scaler=output_scaler,
        squeeze_last_dim=False,
    )


def plot_train_test_loss(train_loss: list, test_loss: list, path: str) -> None:
    """Plot training vs validation loss over epochs."""
    plt.figure()
    plt.plot(range(1, len(train_loss)+1), train_loss, label="Training Loss")
    total_epochs = len(test_loss)
    iters_per_epoch = len(train_loss) // total_epochs
    plt.plot(np.arange(iters_per_epoch, len(train_loss)+1, iters_per_epoch), test_loss, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=300)
    plt.close()


def true_vs_pred_plot(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    """Scatter plot comparing true vs predicted values for each output component."""
    num_outputs = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_outputs))
    for i in range(num_outputs):
        plt.subplot(num_outputs, 1, i + 1)
        plt.scatter(y_true[:, i], y_pred[:, i], s=1, alpha=0.5)
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                 [y_true[:, i].min(), y_true[:, i].max()], 'r--')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Output component {i+1}")
        plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_experiment(
    base_path: Union[str, Path],
    model: GNS,
    train_config: Optional[Union[dict, Any]] = None,
    metrics_dict: Optional[dict] = None,
    input_scaler: Optional[Any] = None,
    output_scaler: Optional[Any] = None,
    extra_files: Optional[Dict[str, Callable[[str], None]]] = None,
) -> None:
    """
    Save experiment artifacts including configs, metrics, scalers and additional files.

    Args:
        base_path: Output directory.
        model: Trained GNS model.
        train_config: Training configuration object or dict.
        metrics_dict: Evaluation metrics.
        input_scaler: Scaler for input features.
        output_scaler: Scaler for output values.
        extra_files: Additional files to save, as {filename: save_function}.
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    if train_config:
        with open(base_path / "training_config.yaml", "w") as f:
            yaml.dump(vars(train_config), f)

    if metrics_dict:
        with open(base_path / "metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)

    if input_scaler:
        with open(base_path / "input_scaler.pkl", "wb") as f:
            pickle.dump(input_scaler, f)
    if output_scaler:
        with open(base_path / "output_scaler.pkl", "wb") as f:
            pickle.dump(output_scaler, f)

    if extra_files:
        for name, save_fn in extra_files.items():
            try:
                save_fn(str(base_path / name))
            except Exception as e:
                print(f"[WARNING] Could not save '{name}': {e}")


def evaluate_model(model: GNS, dataset: pyLOM.NN.Dataset) -> tuple:
    """
    Evaluate a trained model on a dataset.

    Args:
        model: Trained GNS model.
        dataset: Dataset to evaluate.

    Returns:
        Tuple of (targets, predictions, metrics_dict).
    """
    preds = model.predict(dataset)
    targets = dataset[:][1]
    evaluator = RegressionEvaluator()
    evaluator(targets, preds)
    evaluator.print_metrics()
    return targets, preds, evaluator.metrics_dict


def save_full_experiment(
    model: GNS,
    logs: dict,
    targets: np.ndarray,
    preds: np.ndarray,
    config: dict,
    input_scaler: Any,
    output_scaler: Any,
    resudir: Union[str, Path],
) -> None:
    """
    Save the complete experiment including model checkpoint and evaluation artifacts.

    Args:
        model: Trained GNS model.
        logs: Training logs including loss curves.
        targets: True output values.
        preds: Predicted output values.
        config: Training configuration dictionary.
        input_scaler: Input scaler object.
        output_scaler: Output scaler object.
        resudir: Output directory.
    """
    model.save(Path(resudir) / "model.pth")

    save_experiment(
        base_path=resudir,
        model=model,
        train_config=config,
        metrics_dict=logs["metrics"],
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        extra_files={
            "true_vs_pred.png": lambda p: true_vs_pred_plot(targets, preds, p),
            "train_test_loss.png": lambda p: plot_train_test_loss(logs["train_loss"], logs["test_loss"], p),
        }
    )

def get_device_and_seed(config: dict) -> tuple:
    device = config.get("experiment", {}).get("device", "cuda")
    seed = config.get("experiment", {}).get("seed", None)

    if seed is not None:
        set_seed(seed)  # Use pyLOM's set_seed for reproducibility

    return device, seed


def build_model_config(config: dict) -> GNSConfig:
    model_dict = config["model"].copy()
    model_dict["graph_path"] = config["resources"]["graph_path"]
    model_dict["device"] = config["experiment"].get("device", "cuda")
    model_dict["seed"] = config["experiment"].get("seed", None)
    return GNSConfig(**model_dict)

def run_training(config: dict) -> None:
    """Run full training pipeline using a YAML config."""
    resudir = config["execution"]["resudir"]
    os.makedirs(resudir, exist_ok=True)

    device, seed = get_device_and_seed(config)
    config["model"]["device"] = device

    input_scaler = MinMaxScaler()
    output_scaler = None

    td_train = load_dataset(config["resources"]["train_ds"], input_scaler, output_scaler)
    td_val   = load_dataset(config["resources"]["val_ds"], input_scaler, output_scaler)
    td_test  = load_dataset(config["resources"]["test_ds"], input_scaler, output_scaler)

    model_config = build_model_config(config)
    model = GNS(model_config)

    pipeline = Pipeline(
        train_dataset=td_train,
        valid_dataset=td_val,
        test_dataset=td_test,
        model=model,
        training_params=config["training"]
    )

    logs = pipeline.run()
    targets, preds, metrics = evaluate_model(model, td_test)
    logs["metrics"] = metrics

    save_full_experiment(model, logs, targets, preds, config["training"],
                         input_scaler, output_scaler, resudir)

    pyLOM.cr_info()


def run_optuna(config: dict) -> None:
    """Run hyperparameter optimization using Optuna and save best model and results."""
    resudir = config["execution"]["resudir"]
    device, seed = get_device_and_seed(config)
    config["optuna"]["search_space"]["graph"] = Graph.load(config["resources"]["graph_path"], device=device)
    if seed is not None:
        config["optuna"]["search_space"]["model"]["seed"] = seed
    config["optuna"]["search_space"]["model"]["device"] = device

    # Load and scale datasets
    input_scaler = MinMaxScaler()
    output_scaler = None

    td_train = load_dataset(config["resources"]["train_ds"], input_scaler, output_scaler)
    td_val   = load_dataset(config["resources"]["val_ds"], input_scaler, output_scaler)
    td_test  = load_dataset(config["resources"]["test_ds"], input_scaler, output_scaler)

    # Load graph
    graph = Graph.load(config["resources"]["graph_path"])

    # Update Optuna search space to include the graph and optional seed
    search_space = config["optuna"]["search_space"]
    search_space["graph"] = graph
    if seed is not None:
        search_space["model"]["seed"] = seed

    # Configure Optuna optimizer
    optimizer = OptunaOptimizer(
        optimization_params=search_space,
        n_trials=config["optuna"]["study"]["n_trials"],
        direction=config["optuna"]["study"]["direction"],
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config["optuna"]["study"]["pruner"]["n_startup_trials"],
            n_warmup_steps=config["optuna"]["study"]["pruner"]["n_warmup_steps"],
            interval_steps=config["optuna"]["study"]["pruner"]["interval_steps"],
        ),
        save_dir=resudir
    )

    # Create pipeline for optimization
    pipeline = Pipeline(
        train_dataset=td_train,
        valid_dataset=td_val,
        test_dataset=td_test,
        optimizer=optimizer,
        model_class=GNS
    )

    # Run optimization and evaluate best model
    logs = pipeline.run()
    model = pipeline.model
    targets, preds, metrics = evaluate_model(model, td_test)
    logs["metrics"] = metrics

    # Save everything
    save_full_experiment(
        model=model,
        logs=logs,
        targets=targets,
        preds=preds,
        config=model.config,  # <- dataclass already
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        resudir=resudir
    )

    pyLOM.cr_info()


def main() -> None:
    """Entry point for command-line execution."""
    parser = argparse.ArgumentParser(description="Run GNS training or Optuna search.")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    mode = config.get("execution", {}).get("mode", "train")

    if mode == "train":
        run_training(config)
    elif mode == "optuna":
        run_optuna(config)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'train' or 'optuna'.")

    # -------------------------------------------------------------------------
    # Example: Reload the trained model and run inference on dummy input
    # Alternatively, you can use a utility function:
    # demo_inference(config["execution"]["results_dir"], input_tensor)
    # -------------------------------------------------------------------------

    # Step 1: Load the trained model from disk
    model_path = Path(config["execution"]["results_dir"]) / "model.pth"
    device, _ = get_device_and_seed(config)
    model = GNS.load(model_path, device=device)

    model.eval()  # Set to evaluation mode

    # Step 2: Prepare input for inference (AoA=4.0, Mach=0.7)
    dummy_input = np.array([[4.0, 0.7]])  # Shape: (1, 2)
    input_tensor = torch.tensor(dummy_input, dtype=torch.float32, device=device)

    # Step 3: Run prediction
    prediction = model.predict(input_tensor)

    # Step 4: Show output
    print(f"\n>>> Inference output shape: {prediction.shape}")
    print(f">>> Inference output: {prediction.detach().cpu().numpy()}")

if __name__ == "__main__":
    main()
