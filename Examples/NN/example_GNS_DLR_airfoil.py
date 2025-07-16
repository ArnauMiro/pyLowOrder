#!/usr/bin/env python
#
# Refactored GNS Script with new interface
# Last revision: 2025-07-15

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import optuna

import pyLOM
from pyLOM.NN import GNS, MinMaxScaler, OptunaOptimizer, Pipeline, Graph
from pyLOM.NN.utils import save_experiment


def load_yaml_config(path):
    with open(path, 'r') as f:
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

def run_training(basedir, config_path, resudir):
    os.makedirs(resudir, exist_ok=True)

    input_scaler = MinMaxScaler()
    output_scaler = None

    td_train = load_dataset(os.path.join(basedir, 'TRAIN_converter.h5'), input_scaler, output_scaler)
    td_test  = load_dataset(os.path.join(basedir, 'TEST_converter.h5'), input_scaler, output_scaler)
    td_val   = load_dataset(os.path.join(basedir, 'VAL_converter.h5'), input_scaler, output_scaler)

    graph = Graph.load(os.path.join(basedir, 'TRAIN_converter.h5'))

    model, train_cfg = GNS.from_yaml(config_path, with_training_config=True, graph=graph)

    pipeline = Pipeline(
        train_dataset=td_train,
        test_dataset=td_test,
        valid_dataset=td_val,
        model=model,
        training_params=train_cfg
    )

    logs = pipeline.run()

    preds = model.predict(td_test)
    targets = td_test[:][1]

    evaluator = pyLOM.NN.RegressionEvaluator()
    evaluator(targets, preds)
    evaluator.print_metrics()

    # Save all outputs in a structured subfolder
    save_experiment(
        base_path=resudir,
        model=model,
        graph=graph,
        model_config=model.config,
        train_config=train_cfg,
        input_scaler=input_scaler,
        metrics_dict=evaluator.metrics_dict,
        extra_files={
            "true_vs_pred.png": lambda p: true_vs_pred_plot(targets, preds, p),
            "train_test_loss.png": lambda p: plot_train_test_loss(logs['train_loss'], logs['test_loss'], p),
        }
    )

    pyLOM.cr_info()

def run_optuna(basedir, config_path, resudir):
    os.makedirs(resudir, exist_ok=True)

    input_scaler = MinMaxScaler()
    output_scaler = None

    td_train = load_dataset(os.path.join(basedir, 'TRAIN_converter.h5'), input_scaler, output_scaler)
    td_test  = load_dataset(os.path.join(basedir, 'TEST_converter.h5'), input_scaler, output_scaler)
    td_val   = load_dataset(os.path.join(basedir, 'VAL_converter.h5'), input_scaler, output_scaler)

    graph = Graph.load(os.path.join(basedir, 'TRAIN_converter.h5'))

    config = load_yaml_config(config_path)
    optimization_space = config['optuna']
    optimization_space['graph'] = graph

    optimizer = OptunaOptimizer(
        optimization_params=optimization_space,
        n_trials=optimization_space.get('n_trials', 50),
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=10,
            interval_steps=1
        ),
        save_dir=resudir
    )

    pipeline = Pipeline(
        train_dataset=td_train,
        test_dataset=td_test,
        valid_dataset=td_val,
        optimizer=optimizer,
        model_class=GNS
    )

    logs = pipeline.run()

    model = pipeline.model

    preds = model.predict(td_test)
    targets = td_test[:][1]

    evaluator = pyLOM.NN.RegressionEvaluator()
    evaluator(targets, preds)
    evaluator.print_metrics()

    save_experiment(
        base_path=resudir,
        model=model,
        graph=graph,
        model_config=model.config,
        train_config=model.state[5] if hasattr(model, 'state') else None,
        input_scaler=input_scaler,
        metrics_dict=evaluator.metrics_dict,
        extra_files={
            "true_vs_pred.png": lambda p: true_vs_pred_plot(targets, preds, p),
            "train_test_loss.png": lambda p: plot_train_test_loss(logs['train_loss'], logs['test_loss'], p),
        }
    )

    pyLOM.cr_info()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, required=True)
    parser.add_argument('--resudir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'optuna'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        run_training(args.basedir, args.config, args.resudir)
    elif args.mode == 'optuna':
        run_optuna(args.basedir, args.config, args.resudir)

if __name__ == "__main__":
    main()