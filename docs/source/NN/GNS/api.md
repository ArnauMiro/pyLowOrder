# GNS API Reference

This reference covers the key classes and functions involved in training, optimizing, evaluating, and saving Graph Neural Networks (GNS) using `pyLOM.NN`.

---

## 1. `GNS`

**Module:** `pyLOM.NN.GNS`

Main class implementing a Graph Neural Network for regression over CFD fields.

### Constructor

```python
GNS(config: GNSConfig)
```

### Key methods

* `predict(input: torch.Tensor | Dataset) -> np.ndarray`
* `save(path: str) -> None`
* `load(path: str) -> GNS` *(classmethod)*

### Description

A GNS takes node-level input (e.g., AoA, Mach) and predicts quantities like pressure coefficient (`CP`) by propagating through a fixed mesh graph.

**Notes:** The graph is not saved within the model checkpoint — the `graph_path` must always be valid in `config.graph_path`.

---

## 2. `Graph`

**Module:** `pyLOM.NN.Graph`

Graph data structure representing mesh topology.

### Key methods

* `Graph.load(path: str) -> Graph`
* `Graph.save(path: str) -> None`

The graph is required at model initialization and must be created externally.

---

## 3. `Dataset`

**Module:** `pyLOM.NN.Dataset`

Wrapper around CFD datasets that allows:

* Preprocessing (mesh variables, scalers, etc.)
* Batching for training
* Compatibility with the GNS model

### Load Method

```python
Dataset.load(path: str, *, field_names, variables_names, add_variables, add_mesh_coordinates, inputs_scaler, outputs_scaler, squeeze_last_dim) -> Dataset
```

---

## 4. `Pipeline`

**Module:** `pyLOM.NN.Pipeline`

Unifies training or optimization logic for GNS.

### Constructor

```python
Pipeline(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    model: Optional[GNS] = None,
    training_params: Optional[TrainingConfig] = None,
    optimizer: Optional[OptunaOptimizer] = None,
    model_class: Optional[Callable] = None,
)
```

### Method

* `run() -> dict`

  * Runs training loop or Optuna optimization depending on context

---

## 5. `OptunaOptimizer`

**Module:** `pyLOM.NN.OptunaOptimizer`

Hyperparameter tuner for GNS via Optuna.

### Constructor

```python
OptunaOptimizer(
    optimization_params: dict,
    n_trials: int,
    direction: str,
    pruner: optuna.pruners.BasePruner,
    save_dir: Union[str, Path]
)
```

Used inside `Pipeline` when running in optimization mode.

---

## 6. `RegressionEvaluator`

**Module:** `pyLOM.NN.utils`

Provides regression metrics and plots for model validation.

### Methods

* `__call__(y_true, y_pred)`
* `print_metrics()`
* `metrics_dict: dict`

---

## 7. `save_experiment`

**Module:** `pyLOM.NN.utils`

Saves auxiliary experiment artifacts.

```python
def save_experiment(
    base_path: Union[str, Path],
    model: GNS,
    train_config: Optional[dict] = None,
    metrics_dict: Optional[dict] = None,
    input_scaler: Optional[Any] = None,
    output_scaler: Optional[Any] = None,
    extra_files: Optional[Dict[str, Callable[[str], None]]] = None,
) -> None
```

Used after `model.save()` to persist config, metrics, scalers, and plots.

---

For full example usage, see [`usage.md`](usage.md).
For config structure, refer to [`config_reference.md`](config_reference.md).
