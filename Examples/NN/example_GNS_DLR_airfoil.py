import os
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

import pyLOM
from pyLOM.NN import GNS, Graph, MinMaxScaler, OptunaOptimizer, Pipeline
from pyLOM.NN.gns import GNSConfig, TrainingConfig


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_dataset(fname, inputs_scaler, outputs_scaler):
    return pyLOM.NN.Dataset.load(
        fname,
        field_names=["CP"],
        add_variables=True,
        add_mesh_coordinates=False,
        variables_names=["AoA", "Mach"],
        inputs_scaler=inputs_scaler,
        outputs_scaler=outputs_scaler,
        squeeze_last_dim=False,
    )

def save_metrics(metrics_dict, path):
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

def true_vs_pred_plot(y_true, y_pred, path):
    num_plots = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):
        plt.subplot(num_plots, 1, j + 1)
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.5)
        min_val, max_val = y_true[:, j].min(), y_true[:, j].max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"Scatterplot for Component {j+1}")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)

def plot_train_test_loss(train_loss, test_loss, path):
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    total_epochs = len(test_loss)
    total_iters = len(train_loss)
    iters_per_epoch = total_iters // total_epochs
    plt.plot(np.arange(iters_per_epoch, total_iters+1, step=iters_per_epoch), test_loss, label="Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=300)

def save_experiment(resudir, model, config_path, input_scaler, evaluator, logs, run_cfg):
    os.makedirs(resudir, exist_ok=True)

    if run_cfg["evaluation"].get("save_model", True):
        model.save(os.path.join(resudir, run_cfg["evaluation"]["model_path"]))

    input_scaler.save(os.path.join(resudir, "input_scaler.json"))

    if run_cfg["evaluation"].get("save_metrics", True):
        save_metrics(evaluator.metrics_dict, os.path.join(resudir, run_cfg["evaluation"]["metrics_path"]))

    if run_cfg["evaluation"].get("plot_true_vs_pred", True):
        true_vs_pred_plot(evaluator.y_true, evaluator.y_pred, os.path.join(resudir, "true_vs_pred.png"))

    if run_cfg["evaluation"].get("plot_train_test_loss", True):
        plot_train_test_loss(logs["train_loss"], logs["test_loss"], os.path.join(resudir, "train_test_loss.png"))

    # Save YAML config used
    with open(os.path.join(resudir, "config_used.yaml"), "w") as f:
        f.write(Path(config_path).read_text())

def run_training(basedir, resudir, device, run_cfg):
    config_path = run_cfg["train_config"]
    model, training_cfg = GNS.from_yaml(config_path, with_training_config=True)

    input_scaler = MinMaxScaler()
    output_scaler = None

    td_train = load_dataset(os.path.join(basedir, 'TRAIN_converter.h5'), input_scaler, output_scaler)
    td_test  = load_dataset(os.path.join(basedir, 'TEST_converter.h5'), input_scaler, output_scaler)
    td_val   = load_dataset(os.path.join(basedir, 'VAL_converter.h5'), input_scaler, output_scaler)

    pipeline = Pipeline(
        train_dataset=td_train,
        test_dataset=td_test,
        valid_dataset=td_val,
        model=model,
        training_params=training_cfg.__dict__
    )

    logs = pipeline.run()

    preds = model.predict(td_test).cpu()
    labels = td_test[:][1].cpu()

    evaluator = pyLOM.NN.RegressionEvaluator()
    evaluator(labels, preds)
    evaluator.print_metrics()

    evaluator.y_true = labels
    evaluator.y_pred = preds

    save_experiment(resudir, model, config_path, input_scaler, evaluator, logs, run_cfg)
    pyLOM.cr_info()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to run_config.yaml")
    args = parser.parse_args()

    run_cfg = load_yaml_config(args.config)["run_config"]
    basedir = run_cfg["basedir"]
    resudir = run_cfg["resudir"]
    device = pyLOM.NN.select_device("cuda" if torch.cuda.is_available() else "cpu")

    if run_cfg.get("optuna_optimize", False):
        raise NotImplementedError("Optuna mode not yet refactorizado")
    else:
        run_training(basedir, resudir, device, run_cfg)

if __name__ == "__main__":
    main()
