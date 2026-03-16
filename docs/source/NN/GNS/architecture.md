# GNS Architecture

## Runtime components

The GNS training stack is split into these components:

- `Graph` (`pyLOM.NN.gns.graph.Graph`): graph topology and feature dictionaries.
- `GNS` (`pyLOM.NN.architectures.gns.GNS`): message-passing model.
- `Dataset` (`pyLOM.NN.dataset.Dataset`): input/output tensors for training.
- `Pipeline` (`pyLOM.NN.pipeline.Pipeline`): orchestration layer for fixed training or Optuna.
- `GNSTrainingConfig` and nested DTOs: immutable training/runtime configuration.

## Construction path

Typical model construction in scripts:

1. Parse YAML into DTOs via `dacite.from_dict`.
2. Load graph with `GNS.from_graph_path(config=..., graph_path=...)`.
3. Build datasets with `Dataset.load(...)`.
4. Run `Pipeline(...).run()`.

`GNS.__init__` resolves string config entries to runtime objects:

- device from `config.device`
- activation from `config.activation`
- loss/optimizer/scheduler from training config during `fit`

## Graph representation

`Graph` extends `torch_geometric.data.Data` and stores:

- `edge_index` (`[2, E]`)
- `node_features_dict` (named node features)
- `edge_features_dict` (named edge features)
- concatenated tensors `x` and `edge_attr`

`Graph.from_pyLOM_mesh(mesh)` derives features and connectivity from a `pyLOM.Mesh`.

## Training loop

`GNS.fit(train_dataset, eval_dataset, config=...)`:

- validates dataset shape compatibility
- builds input dataloader from `config.dataloader`
- builds subgraph loader from `config.subgraph_loader`
- optionally builds eval loaders when `eval_dataset` is provided
- runs epochs through `_GNSTrainingLoop`
- returns logs including `train_loss` and `test_loss`

When `eval_dataset=None`, no validation loss is produced.

## Reproducibility and persistence

`save_experiment_artifacts(...)` persists:

- model checkpoint (`model.pth`)
- reproducibility config (`config.yaml`)
- metadata (`meta.yaml` with SHA256 fingerprint and git commit)
- metrics and optional extra outputs

For graph persistence, use `Graph.save` / `Graph.load`.
