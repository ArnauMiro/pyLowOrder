# GNS Module Documentation

> **Graph Neural Simulator (GNS)**  
> A specialized module within the `pyLOM` framework for training and evaluating graph-based surrogate models in CFD applications.

---

## Overview

The GNS module is designed to handle **graph-based machine learning models** that simulate physical systems from structured data like computational meshes and boundary conditions. It leverages `pyLOM.NN.GNS` and wraps around a message-passing architecture for surrogate modeling of quantities such as pressure distributions.

This module is part of a larger project but can be used independently for:

- Training a GNS model from scratch using a YAML configuration.
- Running hyperparameter optimization via Optuna.
- Evaluating trained models and visualizing predictions.
- Saving reproducible experiments and checkpoints.
- Reloading trained models for inference or further training.

---

## Key Features

- ✅ **Config-driven training**: Easy to manage experiments via YAML files.
- 📈 **Optuna integration**: Built-in hyperparameter optimization.
- 💾 **Persistent artifacts**: Saves training curves, metrics, scalers, and model weights.
- 🔁 **Inference-ready**: Reload models and predict from new inputs.
- 🧪 **Evaluation tools**: Includes regression metrics and visual diagnostics.

---

## Main Components

| Component         | Description                                                   |
|------------------|---------------------------------------------------------------|
| `GNS`            | Core GNN model class based on message-passing                 |
| `Pipeline`       | Unified training and evaluation loop                          |
| `OptunaOptimizer`| Automated hyperparameter tuning wrapper                       |
| `Dataset.load`   | Scalable loader for CFD and ML-compatible datasets            |
| `save_experiment`| Helper to store configs, metrics, and visuals                 |
| `RegressionEvaluator` | Calculates MSE, MAE, R², etc., and pretty-prints results |

---

## Scripts

| Script         | Purpose                                           |
|----------------|---------------------------------------------------|
| `run_gns.py`   | Entry point for training, tuning, and inference   |

---

## Typical Workflow

1. **Prepare a YAML config** with sections for model, training, resources, and execution.
2. **Run training or optimization** with `run_gns.py`.
3. **Evaluate and save** the model and all relevant outputs.
4. **Reload and infer** with minimal boilerplate.

See the [`usage.md`](usage.md) guide to get started.

---

## Prerequisites

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- `pyLOM` v2 installed  
- Data formatted using `pyLOM.Dataset` interface

---

