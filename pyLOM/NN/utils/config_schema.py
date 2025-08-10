# config_schema.py

from dataclasses import dataclass, field
from typing import Optional, Union, Sequence
import torch
from torch import Tensor


# ----------------------------------------------------------------------
# Base Configs (DTOs puros: strings y primitivos; nada "vivo")
# ----------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class ModelConfigBase:
    """Pure DTO for model configuration.

    - `activation`: import path string of an nn.Module class (e.g., "torch.nn.ReLU").
    - `device`: string like "cpu", "cuda", "cuda:0".
    - `seed`: single source of truth for all RNGs derived at runtime.
    """
    input_dim: int
    output_dim: int
    hidden_size: int
    activation: str = "torch.nn.ReLU"
    p_dropout: float = 0.0
    seed: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, kw_only=True)
class TrainingConfigBase:
    """Pure DTO for training configuration."""
    loss_fn: str = "torch.nn.MSELoss"
    optimizer: str = "torch.optim.Adam"
    scheduler: Optional[str] = "torch.optim.lr_scheduler.StepLR"
    epochs: int = 100
    lr: float = 1e-4
    lr_gamma: float = 0.1
    lr_scheduler_step: int = 1
    print_every: Optional[int] = 1


# ----------------------------------------------------------------------
# GNS-specific Model and Training Configs
# ----------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class GraphSpec:
    """
    Pure DTO describing the graph provenance (not a hyperparameter).

    Notes
    -----
    - `path` is optional because graphs may be constructed in-memory.
    - `id` can be any stable identifier (e.g., dataset version, commit hash).
    """
    path: Optional[str] = None
    id: Optional[str] = None

@dataclass(frozen=True, kw_only=True)
class GNSModelConfig(ModelConfigBase):
    """GNS-specific model configuration (DTO)."""
    latent_dim: int
    num_msg_passing_layers: int
    encoder_hidden_layers: int
    decoder_hidden_layers: int
    message_hidden_layers: int
    update_hidden_layers: int
    groupnorm_groups: int = 2  # verificar divisibilidad en el constructor del modelo


@dataclass(frozen=True, kw_only=True)
class TorchDataloaderConfig:
    """DTO for the main Torch DataLoader settings."""
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: Optional[bool] = None


@dataclass(frozen=True, kw_only=True)
class SubgraphDataloaderConfig:
    """DTO for the subgraph/seed-node DataLoader settings."""
    batch_size: int = 256
    shuffle: bool = True
    input_nodes: Optional[Union[Tensor, Sequence[int]]] = None


@dataclass(frozen=True, kw_only=True)
class GNSTrainingConfig(TrainingConfigBase):
    """Full training configuration for GNS (DTO) with nested dataloaders."""
    dataloader: TorchDataloaderConfig = field(default_factory=TorchDataloaderConfig)
    subgraph_loader: SubgraphDataloaderConfig = field(default_factory=SubgraphDataloaderConfig)
