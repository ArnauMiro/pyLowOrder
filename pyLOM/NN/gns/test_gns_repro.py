# tests/test_gns_repro.py
# -*- coding: utf-8 -*-
"""
Determinism & reproducibility tests for the GNS pipeline.

This suite validates:
  1) Deterministic algorithms enforcement (sanity guard).
  2) Pure forward determinism in eval mode (no training).
  3) One-epoch end-to-end determinism (weights/optimizer/scheduler/loss).
  4) Multi-epoch loss-sequence determinism with shuffling enabled.
  5) Subgraph sampler ordering determinism under shuffle=True.
  6) Contract guard: shuffle=True must receive a torch.Generator (negative test).

Assumptions
-----------
- Your YAML config and referenced datasets are valid and accessible.
- pyLOM imports succeed in the test environment.
- set_seed() is the strengthened version discussed previously.
"""

from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import numpy as np
import torch
import pytest
from dacite import from_dict, Config as DaciteConfig

# Project imports (adjust paths if your package layout differs)
from pyLOM import set_seed
from pyLOM.NN import Dataset, GNS, Pipeline, MinMaxScaler
from pyLOM.NN.utils.config_schema import GNSModelConfig, GNSTrainingConfig
from pyLOM.utils.config_resolvers import load_yaml
from pyLOM.utils import raiseError


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sha256_state_dict(state_dict: dict) -> str:
    """
    Compute a SHA256 digest of a state_dict (model/optimizer/scheduler).

    Tensors are moved to CPU, made contiguous, and hashed by raw bytes.
    Dicts are traversed recursively with key-sorted order for stability.
    """
    hasher = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        hasher.update(k.encode("utf-8"))
        v = state_dict[k]
        if torch.is_tensor(v):
            hasher.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            hasher.update(sha256_state_dict(v).encode("utf-8"))
        else:
            hasher.update(repr(v).encode("utf-8"))
    return hasher.hexdigest()


def clone_state_dict(sd: dict) -> dict:
    """
    Deep-copy a state_dict so assertions do not depend on live tensors/graph.
    """
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
    Strict numerical equality (rtol=0, atol=0). If you expect tiny noise
    (e.g., CPU vs CUDA), relax tolerances here explicitly.
    """
    return torch.allclose(a, b, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfg_paths():
    """
    Locate the example YAML used by the training script.
    """
    config_path = Path("../pyLowOrder/Examples/NN/configs/gns_config.yaml").absolute()
    assert config_path.exists(), f"Config not found: {config_path}"
    return {"config_path": config_path}


@pytest.fixture(scope="session")
def world(cfg_paths):
    """
    Load DTOs, datasets, and scalers once per session to speed up tests.
    """
    cfg = load_yaml(cfg_paths["config_path"])
    dacite_cfg = DaciteConfig(strict=True)

    # DTOs
    model_cfg = from_dict(GNSModelConfig, cfg["model"]["config"], config=dacite_cfg)
    train_cfg = from_dict(GNSTrainingConfig, cfg["training"], config=dacite_cfg)

    # Provenance
    graph_path = cfg["model"]["graph_path"]

    # Datasets + scalers
    inputs_scaler = MinMaxScaler()
    ds_kwargs = dict(
        field_names=["CP"],
        add_variables=True,
        add_mesh_coordinates=False,
        variables_names=["AoA", "Mach"],
        inputs_scaler=inputs_scaler,
        outputs_scaler=None,
        squeeze_last_dim=False,
    )
    ds_train = Dataset.load(cfg["datasets"]["train_ds"], **ds_kwargs)
    ds_val   = Dataset.load(cfg["datasets"]["val_ds"],   **ds_kwargs)
    ds_test  = Dataset.load(cfg["datasets"]["test_ds"],  **ds_kwargs)

    return dict(
        cfg=cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        graph_path=graph_path,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_00_deterministic_algorithms_enforced():
    """
    Sanity guard: torch must be in deterministic-algorithms mode.
    If this fails, non-deterministic ops may be silently allowed.
    """
    set_seed(123)
    assert torch.is_deterministic(), "PyTorch is not in deterministic mode."


def test_01_forward_eval_determinism(world):
    """
    Deterministic inference:
      - Two fresh models with the same seed must produce identical predictions
        on the same dataset (eval mode, no training).
    """
    set_seed(1234)
    m1 = GNS.from_graph_path(config=world["model_cfg"], graph_path=world["graph_path"])
    m1.eval()
    y1 = m1.predict(world["ds_test"]).detach().cpu()

    set_seed(1234)
    m2 = GNS.from_graph_path(config=world["model_cfg"], graph_path=world["graph_path"])
    m2.eval()
    y2 = m2.predict(world["ds_test"]).detach().cpu()

    assert strict_allclose(y1, y2), "Eval predictions differ across identical seeded runs."


def test_02_one_epoch_training_bitwise(world):
    """
    Verify one-epoch determinism end-to-end.

    The test runs two independent, freshly seeded trainings for exactly 1 epoch and asserts:
      - Bitwise-identical model weights (state_dict)
      - Bitwise-identical optimizer state
      - Bitwise-identical scheduler state (or both empty if scheduler is None)
      - Identical final training loss value
    """
    seed = 777

    # Force epochs=1 via dataclasses.replace (no reliance on pipeline overrides).
    cfg_1ep = world["train_cfg"]
    cfg_1ep = cfg_1ep.replace(epochs=1) if hasattr(cfg_1ep, "replace") else replace(cfg_1ep, epochs=1)

    # First run
    set_seed(seed)
    model_a = GNS.from_graph_path(config=world["model_cfg"], graph_path=world["graph_path"])
    pipe_a = Pipeline(
        train_dataset=world["ds_train"],
        valid_dataset=world["ds_val"],
        test_dataset=world["ds_test"],
        model=model_a,
        training_params={"config": cfg_1ep, "reset_state": True},
    )
    logs_a = pipe_a.run()

    sd_a  = clone_state_dict(model_a.state_dict())
    opt_a = clone_state_dict(model_a.optimizer.state_dict())
    sch_a = clone_state_dict(model_a.scheduler.state_dict()) if model_a.scheduler is not None else {}

    h_model_a = sha256_state_dict(sd_a)
    h_opt_a   = sha256_state_dict(opt_a)
    h_sch_a   = sha256_state_dict(sch_a)

    # Second run
    set_seed(seed)
    model_b = GNS.from_graph_path(config=world["model_cfg"], graph_path=world["graph_path"])
    pipe_b = Pipeline(
        train_dataset=world["ds_train"],
        valid_dataset=world["ds_val"],
        test_dataset=world["ds_test"],
        model=model_b,
        training_params={"config": cfg_1ep, "reset_state": True},
    )
    logs_b = pipe_b.run()

    sd_b  = clone_state_dict(model_b.state_dict())
    opt_b = clone_state_dict(model_b.optimizer.state_dict())
    sch_b = clone_state_dict(model_b.scheduler.state_dict()) if model_b.scheduler is not None else {}

    h_model_b = sha256_state_dict(sd_b)
    h_opt_b   = sha256_state_dict(opt_b)
    h_sch_b   = sha256_state_dict(sch_b)

    # Assertions
    assert h_model_a == h_model_b, "Model weights diverged after 1 epoch."
    assert h_opt_a   == h_opt_b,   "Optimizer state diverged after 1 epoch."
    assert h_sch_a   == h_sch_b,   "Scheduler state diverged after 1 epoch."

    # Loss comparison (defensive)
    assert "train_loss" in logs_a and "train_loss" in logs_b, "Missing train_loss in pipeline logs."
    assert len(logs_a["train_loss"]) > 0 and len(logs_b["train_loss"]) > 0, "Empty train_loss sequences."
    la = logs_a["train_loss"][-1]
    lb = logs_b["train_loss"][-1]
    assert np.allclose(la, lb, rtol=0.0, atol=0.0), "Final train loss differs after 1 epoch."


def test_03_multi_epoch_loss_sequence(world):
    """
    Multi-epoch determinism:
      - Train N epochs twice with the same seed and shuffling enabled.
      - Expect identical *sequence* of training (and eval) losses per epoch.
    """
    seed = 999
    epochs = max(3, world["train_cfg"].epochs)  # ensure >= 3

    cfg_N = world["train_cfg"]
    cfg_N = cfg_N.replace(epochs=epochs) if hasattr(cfg_N, "replace") else replace(cfg_N, epochs=epochs)

    def run_once():
        set_seed(seed)
        m = GNS.from_graph_path(config=world["model_cfg"], graph_path=world["graph_path"])
        p = Pipeline(
            train_dataset=world["ds_train"],
            valid_dataset=world["ds_val"],
            test_dataset=world["ds_test"],
            model=m,
            training_params={"config": cfg_N, "reset_state": True},
        )
        logs = p.run()
        return logs["train_loss"], logs.get("test_loss", [])

    train_a, test_a = run_once()
    train_b, test_b = run_once()

    assert len(train_a) == len(train_b) and all(np.isclose(a, b) for a, b in zip(train_a, train_b)), \
        "Train loss sequence differs across identical runs."
    if test_a and test_b:
        assert len(test_a) == len(test_b) and all(np.isclose(a, b) for a, b in zip(test_a, test_b)), \
            "Eval loss sequence differs across identical runs."


def test_04_subgraph_sampler_order(world):
    """
    Subgraph sampler determinism under shuffle=True.

    With a fixed CPU torch.Generator and shuffle=True, the sequence of seed batches
    (observed via seed_mask) must be identical across runs.
    """
    set_seed(2025)
    model = GNS.from_graph_path(config=world["model_cfg"], graph_path=world["graph_path"])

    # Build a subgraph loader with shuffle=True and an explicit generator.
    sg_cfg = world["train_cfg"].subgraph_loader
    sg_cfg = sg_cfg.replace(shuffle=True) if hasattr(sg_cfg, "replace") else replace(sg_cfg, shuffle=True)

    gen1 = torch.Generator(device="cpu").manual_seed(2025)
    loader1 = model._helpers.init_subgraph_loader(sg_cfg, generator=gen1)

    K = min(5, len(loader1))
    first = []
    for i, G in enumerate(loader1):
        first.append(G.seed_mask.detach().cpu().clone())
        if i + 1 >= K:
            break

    # Second pass with the same seed/generator
    set_seed(2025)
    gen2 = torch.Generator(device="cpu").manual_seed(2025)
    loader2 = model._helpers.init_subgraph_loader(sg_cfg, generator=gen2)

    second = []
    for i, G in enumerate(loader2):
        second.append(G.seed_mask.detach().cpu().clone())
        if i + 1 >= K:
            break

    assert len(first) == len(second) == K, "Different number of sampled batches."
    for idx, (a, b) in enumerate(zip(first, second)):
        assert torch.equal(a, b), f"Seed batch #{idx} differs across runs."


def test_05_contract_shuffle_requires_generator(world):
    """
    Contract negative test:
      - The helper should reject shuffle=True when no torch.Generator is provided.
      - This ensures there is no silent RNG source in data order.
    """
    set_seed(31337)
    model = GNS.from_graph_path(config=world["model_cfg"], graph_path=world["graph_path"])

    sg_cfg = world["train_cfg"].subgraph_loader
    sg_cfg = sg_cfg.replace(shuffle=True) if hasattr(sg_cfg, "replace") else replace(sg_cfg, shuffle=True)

    with pytest.raises(Exception):
        # Must raise due to _check_shuffle_generator_contract
        _ = model._helpers.init_subgraph_loader(sg_cfg, generator=None)
