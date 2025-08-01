# GNS Architecture Overview

This document provides an overview of the internal architecture and modular structure used in the GNS (Graph Neural Simulator) pipeline within `pyLOM`.

---

## Overview

The GNS pipeline is designed for modular training, evaluation, and deployment of graph-based surrogate models for CFD simulations.

The system is split into the following logical components:

```
                      +---------------------+
                      |   YAML Config File  |
                      +----------+----------+
                                 |
                      +----------v----------+
                      |   Experiment Script |
                      |    (run_gns.py)     |
                      +----------+----------+
                                 |
          +----------------------+----------------------+
          |                                             |
+---------v--------+                         +----------v----------+
|   Pipeline       |                         |      Graph          |
| (Training/Optuna)|                         | (Topology & Edges) |
+--------+---------+                         +----------+----------+
         |                                              |
+--------v---------+                          +---------v----------+
|      Model       |<------------------------>|   GNSConfig        |
|      (GNS)       |                          +--------------------+
+--------+---------+
         |
+--------v---------+
|     Dataset       |
| (pyLOM Dataset)   |
+------------------+
```

---

## Components

### 1. **Experiment Script** (`run_gns.py`)

* Entry point for launching training or hyperparameter search
* Loads configs, builds model, runs pipeline
* Also performs evaluation and checkpoint saving

### 2. **YAML Config**

* Encodes all training, model, graph, and dataset parameters
* Passed to all subsystems for reproducibility

### 3. **Pipeline** (`pyLOM.NN.Pipeline`)

* Abstraction over training loop and/or Optuna search
* Encapsulates all logic for training and evaluation

### 4. **GNS Model** (`pyLOM.NN.GNS`)

* Implements the graph-based neural network
* Fully driven by a `GNSConfig`
* Contains prediction and checkpoint logic

### 5. **Graph** (`pyLOM.NN.Graph`)

* Stores mesh connectivity: edges, senders, receivers
* Used as fixed structure in GNS
* Must be created externally (e.g., from mesh or preprocessing tool)

### 6. **Dataset** (`pyLOM.NN.Dataset`)

* Handles input/output tensor construction
* Applies scalers, includes mesh/field variables, batching support

---

## Design Notes

* **Separation of Concerns:**

  * Dataset, model, and graph are modular and interchangeable
  * Config-driven execution enables reproducibility and scalability

* **Persistence:**

  * Model and training artifacts are saved using `model.save()` and `save_experiment()`
  * Graph must be saved separately and referenced via `graph_path`

* **Extensibility:**

  * New models, datasets, or optimization strategies can be added by implementing matching interfaces

---

## Training Flow

1. Parse YAML
2. Load Dataset(s) and Graph
3. Build Model from `config["model"]`
4. Run `Pipeline.run()`
5. Evaluate model
6. Save model + logs + artifacts
7. Optionally reload and run inference

---

See [`api.md`](api.md) for full class references.
See [`usage.md`](usage.md) for example usage.
