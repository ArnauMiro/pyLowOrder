# GNS Configuration Reference

This document describes the full structure and options available in the GNS YAML configuration file.

Each training or optimization run requires a configuration file structured into five main sections:

---

## 1. `model`

Defines the model architecture and graph dependencies.

```yaml
model:
  name: GNS                # Model class identifier (required)
  input_dim: 2             # Number of input features per node (e.g., AoA, Mach)
  output_dim: 1            # Number of output features per node (e.g., CP)
  hidden_dim: 64           # Size of hidden representations
  message_passing_steps: 10  # Number of message passing iterations
  graph_path: /path/to/graph.pth  # Required: path to the graph file
```

**Notes:**

* `graph_path` must point to a serialized `Graph` object created with `pyLOM`.

---

## 2. `training`

Specifies hyperparameters for model training.

```yaml
training:
  optimizer: adam      # Optimizer name ('adam', 'sgd', etc.)
  lr: 0.001             # Learning rate
  batch_size: 128       # Batch size
  epochs: 200           # Number of training epochs
  weight_decay: 0.0     # Optional L2 regularization
  scheduler: null       # Optional learning rate scheduler
```

**Optional fields:**

* `weight_decay` (float): L2 regularization term
* `scheduler`: Learning rate schedule (not yet implemented)

---

## 3. `resources`

Paths to required data and graph assets.

```yaml
resources:
  train_ds: /path/to/train.pkl     # Required
  val_ds: /path/to/val.pkl         # Required
  test_ds: /path/to/test.pkl       # Required
  graph_path: /path/to/graph.pth   # Required
```

**Important:**

* All datasets must be compatible with `pyLOM.NN.Dataset.load()`
* Graph is required and must be serialized

---

## 4. `execution`

Controls execution behavior of the experiment.

```yaml
execution:
  mode: train                # "train" or "optuna"
  results_dir: ./results/    # Output folder for artifacts
```

**Options:**

* `mode`: Selects whether to run a fixed training (`train`) or hyperparameter optimization (`optuna`).
* `results_dir`: Destination for outputs

---

## 5. `optuna`

Optional. Defines hyperparameter optimization strategy using Optuna.
Only used when `execution.mode: optuna`.

```yaml
optuna:
  search_space:
    hidden_dim: [32, 64, 128]  # Integer categorical values
    lr: [1e-4, 1e-3, 5e-3]     # Learning rates to try
  study:
    n_trials: 20              # Number of Optuna trials
    direction: minimize       # Objective to minimize (usually loss)
    pruner:
      n_startup_trials: 5
      n_warmup_steps: 10
      interval_steps: 5
```

**Tips:**

* `search_space` keys must match arguments in the model constructor or training config
* Use `optuna.trial.suggest_categorical()` internally

---

## Example Minimal Configuration

```yaml
model:
  name: GNS
  input_dim: 2
  output_dim: 1
  hidden_dim: 64
  message_passing_steps: 10
  graph_path: ./assets/graph.pth

training:
  optimizer: adam
  lr: 0.001
  batch_size: 128
  epochs: 100

resources:
  train_ds: ./data/train.pkl
  val_ds: ./data/val.pkl
  test_ds: ./data/test.pkl
  graph_path: ./assets/graph.pth

execution:
  mode: train
  results_dir: ./results/demo_run
```

---

For a step-by-step walkthrough, continue to [`usage.md`](usage.md).
To learn about key interfaces, visit [`api.md`](api.md).
