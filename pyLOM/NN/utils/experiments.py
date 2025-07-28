#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling
#
# experiments.py — NN utilities for saving experiment artifacts
#
# Provides standardized utilities to save all essential artifacts
# of a model training run in a reproducible and traceable format.
#
# This includes:
#   • YAML dump of the model configuration (from model.config)
#   • Metrics as JSON
#   • Scalers as pickle objects
#   • Auxiliary outputs (plots, etc.)
#   • Metadata file with config hash, description, and tags
#
# The saved config.yaml is for human readability and documentation only.
# Model loading is based exclusively on the configuration embedded in the .pth checkpoint.
#
# Last rev: 27/07/2025


import os
import json
import yaml
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, Callable, Any


def save_experiment(
    base_path: Path,
    model,
    full_config: dict,
    metrics_dict: dict,
    input_scaler=None,
    output_scaler=None,
    extra_files: dict = None,
):
    """
    Save a complete experiment in a reproducible and structured way.
    Includes model checkpoint, full YAML config, scalers, metrics, and metadata.
    Generates a meta.yaml for easy parsing and reproducibility.

    Args:
        base_path (Path): Where to save the experiment.
        model: Trained model (must have .save() and .config).
        full_config (dict): Full configuration dictionary (as parsed from YAML).
        metrics_dict (dict): Evaluation results.
        input_scaler (optional): Fitted input scaler (e.g., MinMaxScaler).
        output_scaler (optional): Fitted output scaler.
        extra_files (dict): Mapping {filename: generator(path)} to save additional assets (e.g., plots).
    """

    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────
    # Save model checkpoint
    # ─────────────────────────────────────────────────────
    model_path = base_path / "model.pth"
    model.save(model_path)

    # ─────────────────────────────────────────────────────
    # Save full YAML configuration
    # ─────────────────────────────────────────────────────
    config_path = base_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(full_config, f)

    # Compute SHA-1 of the config
    config_hash = hashlib.sha1(config_path.read_bytes()).hexdigest()

    # ─────────────────────────────────────────────────────
    # Save scalers
    # ─────────────────────────────────────────────────────
    scaler_files = []
    if input_scaler is not None:
        fpath = base_path / "input_scaler.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(input_scaler, f)
        scaler_files.append("input_scaler.pkl")

    if output_scaler is not None:
        fpath = base_path / "output_scaler.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(output_scaler, f)
        scaler_files.append("output_scaler.pkl")

    # ─────────────────────────────────────────────────────
    # Save evaluation metrics
    # ─────────────────────────────────────────────────────
    metrics_path = base_path / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics_dict, f)

    # ─────────────────────────────────────────────────────
    # Save extra generated files (e.g., plots)
    # ─────────────────────────────────────────────────────
    extra_files_saved = []
    if extra_files:
        for name, generator in extra_files.items():
            path = base_path / name
            generator(path)
            extra_files_saved.append(name)

    # ─────────────────────────────────────────────────────
    # Save meta.yaml
    # ─────────────────────────────────────────────────────
    meta = {
        "experiment_name": full_config.get("experiment", {}).get("name", "unnamed_experiment"),
        "saved_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "config_hash": config_hash,
        "files": (
            ["model.pth", "config.yaml", "metrics.yaml"]
            + scaler_files
            + extra_files_saved
        )
    }

    meta_path = base_path / "meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)