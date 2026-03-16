# GNS Configuration Reference

This reference describes the configuration schema currently used by GNS scripts in this repository.

## Top-level sections

Expected top-level keys:

- `experiment`
- `datasets`
- `dataset_config`
- `model`
- `training`
- `optuna` (optional)

## `experiment`

Typical fields:

- `name`: run name
- `results_path`: output directory
- `show_plots`: whether to display plots in interactive scripts
- `mode`: optional, commonly `train` or `optuna` in larger scripts

## `datasets`

Paths to dataset files:

- `train_ds`: required
- `test_ds`: required for evaluation scripts
- `val_ds`: optional (`null` allowed)

## `dataset_config`

Used to build `Dataset.load(...)` arguments.

Common fields:

- `field_names` (list[str]): output fields to learn
- `variables_names` (list[str]): variable names used as inputs/parameters
- `add_variables` (bool)
- `add_mesh_coordinates` (bool)
- `mesh_shape` (list[int])
- `scale_inputs` / `scale_outputs` (bool)
- optional scaler type controls in advanced scripts

## `model`

```yaml
model:
  graph_path: /path/to/graph.h5
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
    debug: false
```

`model.config` maps to `GNSModelConfig`.

## `training`

`training` maps to `GNSTrainingConfig`.

Core fields:

- `loss_fn`, `optimizer`, `scheduler` (import path strings)
- `epochs`, `lr`, `weight_decay`, `lr_gamma`, `lr_scheduler_step`
- `print_every`
- `best_metric`, `best_metric_space`
- `weighted_loss_alpha`
- `nan_guard_enabled`
- gradient clipping fields

Nested sections:

- `dataloader`
  - `batch_size`, `shuffle`, `num_workers`, `pin_memory`
- `subgraph_loader`
  - `batch_size`, `shuffle`, `input_nodes`, `mode`
  - `seed_selector` with:
    - `type`: `all` | `auto_frac` | `explicit_list`
    - `frac`
    - `nodes_path`

## `optuna` (optional)

For optimization scripts, an optional `optuna` section may include:

- `study` settings (`n_trials`, `direction`, sampler/pruner config)
- `optimization_params` search space

The exact structure is script-defined, while model/training DTO validation still applies to instantiated configs.
