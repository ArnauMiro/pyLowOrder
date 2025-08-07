#!/usr/bin/env python
#
# pyLOM, NN
#
# Dataclass configurations for GNS model training.
#
# Last rev: 04/08/2025

from dataclasses import dataclass, field
from typing import Optional, Union, Any, Sequence
import torch
from torch import Tensor



#----------------------------------------------------------------------
# Base dataclass configurations
#----------------------------------------------------------------------

@dataclass(kw_only=True)
class ModelConfigBase:
    input_dim: int
    output_dim: int
    hidden_size: int
    activation: type[torch.nn.Module]
    p_dropout: float = 0.0
    seed: Optional[int] = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TorchDataloaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: Optional[bool] = None  # let helper decide default
    generator: Optional[torch.Generator] = None


@dataclass
class TrainingConfigBase:
    loss_fn: Any = torch.nn.MSELoss(reduction='mean')
    optimizer: Any = torch.optim.Adam
    scheduler: Any = torch.optim.lr_scheduler.StepLR
    epochs: int = 100
    lr: float = 1e-4
    lr_gamma: float = 0.1
    lr_scheduler_step: int = 1
    print_every: int = 1





#----------------------------------------------------------------------
# Model-specific configurations (Add KAN, MLP, etc. as needed)
#----------------------------------------------------------------------

@dataclass
class GNSModelConfig(ModelConfigBase):
    latent_dim: int
    num_msg_passing_layers: int
    encoder_hidden_layers: int
    decoder_hidden_layers: int
    message_hidden_layers: int
    update_hidden_layers: int
    groupnorm_groups: int = 2

    def __post_init__(self):
        if self.latent_dim % self.groupnorm_groups != 0:
            raise ValueError(
                f"'latent_dim' ({self.latent_dim}) must be divisible by 'groupnorm_groups' ({self.groupnorm_groups})."
            )

@dataclass
class SubgraphDataloaderConfig:
    batch_size: int = 256
    shuffle: bool = True
    input_nodes: Optional[Union[Tensor, Sequence[int]]] = None
    generator: Optional[torch.Generator] = None



@dataclass
class GNSTrainingConfig(TrainingConfigBase):
    dataloader: TorchDataloaderConfig = field(default_factory=TorchDataloaderConfig)
    subgraph_loader: SubgraphDataloaderConfig = field(default_factory=SubgraphDataloaderConfig)
