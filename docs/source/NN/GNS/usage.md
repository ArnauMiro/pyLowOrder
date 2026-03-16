# GNS Usage

This page shows the current workflow used in this repository for GNS training.

## 1) Prepare inputs

You need:

- a graph file (`.h5`, `.pt`, or `.pkl`) serialized with `Graph.save`
- train/test datasets in pyLOM dataset format (`.h5`)
- a YAML config with the sections expected by the training script

Minimal example files in this repository:

- `Examples/NN/example_GNS_DLR_airfoil.py`
- `Examples/NN/example_GNS_cylinder_dataset.py`
- `Examples/NN/configs/example_GNS_DLR_airfoil_config.yaml`
- `Examples/NN/configs/example_GNS_cylinder_dataset_config.yaml`

## 2) Configuration layout

Current scripts use this top-level structure:

```yaml
experiment:
  name: "gns_cylinder_minimal"
  results_path: "..."
  show_plots: true

datasets:
  train_ds: "..."
  test_ds: "..."
  val_ds: null

dataset_config:
  field_names: ["VELOX"]
  variables_names: ["time"]
  add_variables: true
  add_mesh_coordinates: false
  mesh_shape: [89351]
  scale_inputs: false
  scale_outputs: false

model:
  graph_path: "..."
  config:
    input_dim: 1
    output_dim: 1
    hidden_size: 128
    latent_dim: 16
    num_msg_passing_layers: 1
    encoder_hidden_layers: 2
    decoder_hidden_layers: 1
    message_hidden_layers: 1
    update_hidden_layers: 1
    groupnorm_groups: 1
    activation: "torch.nn.ELU"
    p_dropout: 0.0
    seed: 42
    device: "cpu"

training:
  loss_fn: "torch.nn.MSELoss"
  optimizer: "torch.optim.Adam"
  scheduler: "torch.optim.lr_scheduler.StepLR"
  epochs: 20
  lr: 1.0e-3
  weight_decay: 0.0
  lr_gamma: 0.98
  lr_scheduler_step: 1
  print_every: 1
  dataloader:
    batch_size: 4
    shuffle: true
    num_workers: 0
    pin_memory: false
  subgraph_loader:
    batch_size: 2048
    shuffle: true
    input_nodes: null
    mode: "nodes"
    seed_selector:
      type: "all"
      frac: null
      nodes_path: null
```

## 3) Train

Example command:

```bash
python Examples/NN/example_GNS_DLR_airfoil.py
```

The script:

- loads config with `load_yaml`
- builds `Dataset` objects with `Dataset.load`
- creates model with `GNS.from_graph_path`
- trains with `Pipeline(...).run()`
- evaluates with `RegressionEvaluator`
- saves artifacts through `save_experiment_artifacts`

## 4) Validation dataset behavior

`Pipeline` does not automatically reuse test as validation in normal training mode.

- If `valid_dataset=None`, `GNS.fit(..., eval_dataset=None)` runs without validation loss.
- In Optuna mode, when validation is missing, `Pipeline` falls back to the training dataset and emits a warning.

## 5) Artifacts

`save_experiment_artifacts` writes (at least):

- `model.pth`
- `config.yaml` (`repro_config` block)
- `meta.yaml` (including config SHA256 and git commit)
- `metrics.yaml`
- optional scalers (`inputs_scaler.json`, `outputs_scaler.json`)
- any extra files passed via `extra_files`

See also: `api.md`.
