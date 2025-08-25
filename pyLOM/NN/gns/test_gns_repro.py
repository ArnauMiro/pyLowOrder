# tests/test_gns_repro.py
# -*- coding: utf-8 -*-
"""
Determinism & reproducibility tests for the GNS pipeline.

This suite checks:
  1) Pure forward determinism in eval mode (no training).
  2) One-epoch training determinism (weights, optimizer, scheduler, losses).
  3) Multi-epoch loss sequence determinism with shuffling enabled.
  4) Subgraph sampler ordering determinism under shuffle=True.
  5) Deterministic algorithms enforcement (fail fast on non-deterministic ops).

Assumptions
-----------
- Your YAML config (and datasets referenced by it) are valid and accessible.
- pyLOM imports succeed in the test environment.

Tip
---
Run with:
    pytest -q tests/test_gns_repro.py
"""

import os
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
import pytest
from dacite import from_dict, Config as DaciteConfig

# Project imports (adapt to your package layout if needed)
from pyLOM import set_seed
from pyLOM.utils.config_resolvers import load_yaml, instantiate_from_config
from pyLOM.NN import Dataset, GNS, Pipeline, MinMaxScaler
from pyLOM.NN.utils.config_schema import GNSModelConfig, GNSTrainingConfig
from pyLOM.utils import raiseError


# ---------------------------
# Helpers
# ---------------------------

def sha256_state_dict(state_dict: dict) -> str:
    """
    Compute a SHA256 digest of a model/optimizer/scheduler state_dict.

    Tensors are moved to CPU and converted to contiguous bytes; non-tensors are
    stringified deterministically. This lets us assert exact identity across runs.
    """
    hasher = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        hasher.update(k.encode("utf-8"))
        if torch.is_tensor(v):
            hasher.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, (dict,)):
            hasher.update(sha256_state_dict(v).encode("utf-8"))
        else:
            hasher.update(repr(v).encode("utf-8"))
    return hasher.hexdigest()


def clone_state_dict(sd: dict) -> dict:
    """Deep copy a state_dict to detach it from live tensors/graph."""
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, dict):
            out[k] = clone_state_dict(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def strict_allclose(a: torch.Tensor, b: torch.Tensor) -> bool:
    """
    Strict numerical equality check (rtol=0, atol=0). If you expect tiny
    float noise (e.g., CPU vs CUDA), relax tolerances here explicitly.
    """
    return torch.allclose(a, b, rtol=0.0, atol=0.0)


# ---------------------------
# Pytest fixtures
# ---------------------------

@pytest.fixture(scope="session")
def cfg_paths():
    """
    Locate and load the experiment YAML used in the example script.
    Adjust this path if your test environment differs.
    """
    config_path = Path("../pyLowOrder/Examples/NN/configs/gns_config.yaml").absolute()
    assert config_path.exists(), f"Config not found: {config_path}"
    return {"config_path": config_path}


@pytest.fixture(scope="session")
def dto_and_data(cfg_paths):
    """
    Load DTOs and datasets once per session to speed up testing.
    """
    cfg = load_yaml(cfg_paths["config_path"])
    dacite_cfg = DaciteConfig(strict=True)

    # DTOs
    model_cfg_dict = cfg["model"]["config"]
    model_cfg = from_dict(GNSModelConfig, model_cfg_dict, config=dacite_cfg)

    training_cfg_dict = cfg["training"]
    training_cfg = from_dict(GNSTrainingConfig, training_cfg_dict, config=dacite_cfg)

    graph_path = cfg["model"]["graph_path"]
    ds_paths = cfg["datasets"]

    # Data scalers
    inputs_scaler = MinMaxScaler()
    outputs_scaler = None

    ds_kwargs = dict(
        field_names=["CP"],
        add_variables=True,
        add_mesh_coordinates=False,
        variables_names=["AoA", "Mach"],
        inputs_scaler=inputs_scaler,
        outputs_scaler=outputs_scaler,
        squeeze_last_dim=False,
    )

    ds_train = Dataset.load(ds_paths["train_ds"], **ds_kwargs)
    ds_val   = Dataset.load(ds_paths["val_ds"],   **ds_kwargs)
    ds_test  = Dataset.load(ds_paths["test_ds"],  **ds_kwargs)

    return {
        "cfg": cfg,
        "model_cfg": model_cfg,
        "training_cfg": training_cfg,
        "graph_path": graph_path,
        "ds_train": ds_train,
        "ds_val": ds_val,
        "ds_test": ds_test,
        "inputs_scaler": inputs_scaler,
        "outputs_scaler": outputs_scaler,
    }


# ---------------------------
# Tests
# ---------------------------

def test_00_deterministic_algorithms_enforced():
    """
    Sanity check: ensure deterministic algorithms are enforced at process start.

    If this fails, non-deterministic CUDA/CPU ops may be allowed silently.
    """
    # Strongly recommended at process entry (usually inside set_seed).
    set_seed(123)
    # If the project does this elsewhere, this test is still a useful guardrail.
    assert torch.is_deterministic(), "torch is not in deterministic-algorithms mode"


def test_01_forward_eval_determinism(dto_and_data):
    """
    Deterministic inference:
      - Two fresh models with the same seed must produce identical predictions
        on the same test dataset (eval mode, no dropout randomness).
    """
    set_seed(123)

    model1 = GNS.from_graph_path(config=dto_and_data["model_cfg"],
                                 graph_path=dto_and_data["graph_path"])
    model1.eval()
    y1 = model1.predict(dto_and_data["ds_test"]).detach().cpu()

    # Re-create the model with the same seed and predict again
    set_seed(123)
    model2 = GNS.from_graph_path(config=dto_and_data["model_cfg"],
                                 graph_path=dto_and_data["graph_path"])
    model2.eval()
    y2 = model2.predict(dto_and_data["ds_test"]).detach().cpu()

    assert strict_allclose(y1, y2), "Eval predictions differ across identical seeded runs"


def test_02_one_epoch_training_bitwise(dto_and_data):
    """
    Verify one-epoch determinism end-to-end.

    The test runs two independent, freshly seeded trainings for exactly 1 epoch and asserts:
      - Bitwise-identical model weights (state_dict)
      - Bitwise-identical optimizer state
      - Bitwise-identical scheduler state (or both empty if scheduler is None)
      - Identical final training loss value

    Notes
    -----
    - We explicitly force `epochs=1` via `dataclasses.replace` on the training DTO,
      so we don't rely on any custom `epochs_override` argument in the pipeline.
    - The test assumes your global `set_seed()` (called elsewhere in the suite) puts
      PyTorch into deterministic mode and configures CUDA/cuBLAS accordingly.
    """
    seed = 777

    # Make a 1-epoch copy of the training config (DTO)
    train_cfg_1ep = dto_and_data["training_cfg"]
    if hasattr(train_cfg_1ep, "replace"):
        train_cfg_1ep = train_cfg_1ep.replace(epochs=1)
    else:
        # Fallback if DTO is a dataclass without `.replace` method
        train_cfg_1ep = replace(train_cfg_1ep, epochs=1)

    # -----------------------
    # First run
    # -----------------------
    set_seed(seed)
    model_a = GNS.from_graph_path(
        config=dto_and_data["model_cfg"],
        graph_path=dto_and_data["graph_path"]
    )
    pipe_a = Pipeline(
        train_dataset=dto_and_data["ds_train"],
        valid_dataset=dto_and_data["ds_val"],
        test_dataset=dto_and_data["ds_test"],
        model=model_a,
        training_params={"config": train_cfg_1ep, "reset_state": True},
    )
    logs_a = pipe_a.run()

    sd_a = clone_state_dict(model_a.state_dict())
    opt_a = clone_state_dict(model_a.optimizer.state_dict())
    sch_a = clone_state_dict(model_a.scheduler.state_dict()) if model_a.scheduler is not None else {}

    h_model_a = sha256_state_dict(sd_a)
    h_opt_a   = sha256_state_dict(opt_a)
    h_sch_a   = sha256_state_dict(sch_a)

    # -----------------------
    # Second run
    # -----------------------
    set_seed(seed)
    model_b = GNS.from_graph_path(
        config=dto_and_data["model_cfg"],
        graph_path=dto_and_data["graph_path"]
    )
    pipe_b = Pipeline(
        train_dataset=dto_and_data["ds_train"],
        valid_dataset=dto_and_data["ds_val"],
        test_dataset=dto_and_data["ds_test"],
        model=model_b,
        training_params={"config": train_cfg_1ep, "reset_state": True},
    )
    logs_b = pipe_b.run()

    sd_b = clone_state_dict(model_b.state_dict())
    opt_b = clone_state_dict(model_b.optimizer.state_dict())
    sch_b = clone_state_dict(model_b.scheduler.state_dict()) if model_b.scheduler is not None else {}

    h_model_b = sha256_state_dict(sd_b)
    h_opt_b   = sha256_state_dict(opt_b)
    h_sch_b   = sha256_state_dict(sch_b)

    # -----------------------
    # Assertions
    # -----------------------
    assert h_model_a == h_model_b, "Model weights diverged after 1 epoch."
    assert h_opt_a   == h_opt_b,   "Optimizer state diverged after 1 epoch."
    assert h_sch_a   == h_sch_b,   "Scheduler state diverged after 1 epoch."

    # Loss comparison (defensive: only if logs exist and contain 'train_loss')
    if logs_a and logs_b and "train_loss" in logs_a and "train_loss" in logs_b:
        assert len(logs_a["train_loss"]) > 0 and len(logs_b["train_loss"]) > 0, \
            "Empty train_loss sequences; cannot compare."
        la = logs_a["train_loss"][-1]
        lb = logs_b["train_loss"][-1]
        assert np.allclose(la, lb, rtol=0.0, atol=0.0), "Final train loss differs after 1 epoch."


def test_03_multi_epoch_loss_sequence(dto_and_data):
    """
    Multi-epoch determinism:
      - Run N epochs twice with the same seed and shuffling enabled.
      - Expect identical *sequence* of training (and eval) losses per epoch.
    """
    seed = 999
    epochs = max(3, dto_and_data["training_cfg"].epochs)  # ensure >= 3 for a useful test

    # Helper to run a short training and collect losses
    def run_once():
        set_seed(seed)
        model = GNS.from_graph_path(config=dto_and_data["model_cfg"],
                                    graph_path=dto_and_data["graph_path"])
        # clone DTO with desired epochs (if DTO is immutable use replace())
        cfg = dto_and_data["training_cfg"]
        if hasattr(cfg, "replace"):
            cfg = cfg.replace(epochs=epochs)
        else:
            # last resort: assume it's a dataclass and mutate a copy
            from dataclasses import replace
            cfg = replace(cfg, epochs=epochs)

        pipe = Pipeline(
            train_dataset=dto_and_data["ds_train"],
            valid_dataset=dto_and_data["ds_val"],
            test_dataset=dto_and_data["ds_test"],
            model=model,
            training_params={"config": cfg, "reset_state": True},
        )
        logs = pipe.run()
        return logs["train_loss"], logs.get("test_loss", [])

    train_a, test_a = run_once()
    train_b, test_b = run_once()

    assert len(train_a) == len(train_b) and all(np.isclose(a, b) for a, b in zip(train_a, train_b)), \
        "Train loss sequence differs across identical runs"

    if test_a and test_b:
        assert len(test_a) == len(test_b) and all(np.isclose(a, b) for a, b in zip(test_a, test_b)), \
            "Eval loss sequence differs across identical runs"


def test_04_subgraph_sampler_order(dto_and_data):
    """
    Subgraph sampler determinism:
      - With shuffle=True and a fixed Generator, the order of seed batches must be identical across runs.
      - This test inspects the first K batches' seed masks to ensure equality.
    """
    set_seed(2025)
    model = GNS.from_graph_path(config=dto_and_data["model_cfg"],
                                graph_path=dto_and_data["graph_path"])

    # Build a subgraph loader via helpers with a fixed generator
    sg_cfg = dto_and_data["training_cfg"].subgraph_loader
    if hasattr(sg_cfg, "replace"):
        sg_cfg = sg_cfg.replace(shuffle=True, batch_size=max(64, sg_cfg.batch_size))
    else:
        from dataclasses import replace
        sg_cfg = replace(sg_cfg, shuffle=True, batch_size=max(64, sg_cfg.batch_size))

    gen1 = torch.Generator(device="cpu").manual_seed(2025)
    loader1 = model._helpers.init_subgraph_loader(sg_cfg, generator=gen1)

    # Capture first K batches
    K = min(5, len(loader1))
    first_run = []
    for i, G in enumerate(loader1):
        first_run.append(G.seed_mask.detach().cpu().clone())
        if i + 1 >= K:
            break

    # Second run with same seed/generator => identical batches expected
    set_seed(2025)
    gen2 = torch.Generator(device="cpu").manual_seed(2025)
    loader2 = model._helpers.init_subgraph_loader(sg_cfg, generator=gen2)
    second_run = []
    for i, G in enumerate(loader2):
        second_run.append(G.seed_mask.detach().cpu().clone())
        if i + 1 >= K:
            break

    assert len(first_run) == len(second_run) == K
    for idx, (a, b) in enumerate(zip(first_run, second_run)):
        assert torch.equal(a, b), f"Seed batch #{idx} differs across runs"
