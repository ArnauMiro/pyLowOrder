# GNS API Reference (Practical)

This page summarizes the public interfaces used by current repository scripts.

## `pyLOM.NN.architectures.gns.GNS`

Main constructors:

- `GNS.from_graph(config, graph: Graph)`
- `GNS.from_graph_path(config, graph_path: str | Path)`

Main methods:

- `fit(train_dataset, eval_dataset=None, config, reset_state=True, ...)`
- `predict(X, config=None, ...)`
- `save(path)`
- `load(path, device=...)`

Notes:

- The model stores graph provenance via `graph_spec`.
- `fit` returns logs including `train_loss` and `test_loss`.

## `pyLOM.NN.gns.graph.Graph`

Key constructors/methods:

- `Graph.from_pyLOM_mesh(mesh, device=None)`
- `Graph.save(fname, mode=None)`
- `Graph.load(fname, device=...)`

Feature storage:

- `node_features_dict`
- `edge_features_dict`
- concatenated `x` / `edge_attr`

## `pyLOM.NN.pipeline.Pipeline`

Constructor:

```python
Pipeline(
    train_dataset,
    valid_dataset=None,
    test_dataset=None,
    model=None,
    training_params=None,
    optimizer=None,
    model_class=None,
)
```

Behavior:

- fixed-training mode: requires `model` + `training_params`
- optuna mode: requires `optimizer` + `model_class`
- in optuna mode, if `valid_dataset` is missing, it falls back to `train_dataset`

## Experiment helpers (`pyLOM.NN.utils.experiment`)

Most-used function:

- `save_experiment_artifacts(...)`

Companion plotting utilities commonly used in scripts:

- `plot_training_and_validation_loss(...)`
- `plot_train_test_loss(...)`
- `plot_true_vs_pred(...)`

## Evaluators and datasets

- `pyLOM.NN.Dataset.load(...)`
- `pyLOM.NN.utils.RegressionEvaluator`

For generated API docs with full signatures, see `docs/source/api/pyLOM.NN*.rst` pages.
