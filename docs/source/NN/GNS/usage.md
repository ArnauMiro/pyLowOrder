# GNS Module Usage Guide

This guide walks through how to train, optimize, evaluate, and use a GNS model using the `pyLOM` framework.

---

## 1. Prepare Configuration File

Create a YAML configuration file (`config.yaml`) that includes the following sections:

```yaml
model:
  name: GNS
  input_dim: 2
  output_dim: 1
  hidden_dim: 64
  message_passing_steps: 10
  graph_path: /path/to/graph.pth

training:
  optimizer: adam
  lr: 0.001
  batch_size: 128
  epochs: 200

resources:
  train_ds: /path/to/train_dataset.pkl
  val_ds: /path/to/val_dataset.pkl
  test_ds: /path/to/test_dataset.pkl
  graph_path: /path/to/graph.pth

execution:
  mode: train  # or 'optuna'
  results_dir: ./results/gns_run_01

optuna:
  search_space:
    hidden_dim: [32, 64, 128]
    lr: [1e-4, 5e-4, 1e-3]
  study:
    n_trials: 20
    direction: minimize
    pruner:
      n_startup_trials: 5
      n_warmup_steps: 10
      interval_steps: 5
```

---

## 2. Run Training

Use the command-line entry point to train the model:

```bash
python run_gns.py --config config.yaml
```

This will:

* Load datasets and graph
* Initialize the model
* Train using the `Pipeline` abstraction
* Evaluate and save metrics
* Export plots and model checkpoint

---

## 3. Run Hyperparameter Optimization

Change `execution.mode` in the YAML to `optuna`, then execute:

```bash
python run_gns.py --config config.yaml
```

This runs Optuna trials, logs results, and saves the best-performing model.

---

## 4. Inference Example

After training, a GNS model checkpoint is saved to `results_dir/model.pth`. To perform inference:

```python
from pyLOM.NN import GNS
import torch
import numpy as np

model = GNS.load("results/gns_run_01/model.pth")
model.eval()

# Example input: AoA=4.0, Mach=0.7
input_np = np.array([[4.0, 0.7]], dtype=np.float32)
input_tensor = torch.tensor(input_np)
output = model.predict(input_tensor)

print("Prediction shape:", output.shape)
print("Prediction values:", output.detach().cpu().numpy())
```

---

## 5. Output Artifacts

Each run produces the following outputs in `results_dir`:

* `model.pth`: Model weights + config + state
* `training_config.yaml`: The config used for training
* `metrics.json`: Evaluation metrics (MSE, MAE, R², etc.)
* `input_scaler.pkl`, `output_scaler.pkl`: Scaling objects (if used)
* `train_test_loss.png`: Training vs validation loss
* `true_vs_pred.png`: Scatter plots of predictions vs ground truth

---

## 6. Notes

* Ensure that all required paths in the YAML config are valid.
* The graph must be saved to disk and referenced via `graph_path`.
* Use `MinMaxScaler` or other scaler classes for consistent preprocessing.
* For continued training, load the model via `GNS.load()` and pass to a new `Pipeline`.

---

Continue to [`config_reference.md`](config_reference.md) for details on all supported config fields.
