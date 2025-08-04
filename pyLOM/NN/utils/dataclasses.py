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

@dataclass
class ModelConfig:
    input_dim: int
    output_dim: int
    seed: Optional[int] = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DataloaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 1
    pin_memory: bool = True
    generator: Optional[torch.Generator] = None


@dataclass
class TrainingConfig:
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
class GNSModelConfig(ModelConfig):
    latent_dim: int
    hidden_size: int
    num_msg_passing_layers: int
    encoder_hidden_layers: int
    decoder_hidden_layers: int
    message_hidden_layers: int
    update_hidden_layers: int
    groupnorm_groups: int = 2
    p_dropouts: float = 0.0
    activation: torch.nn.Module = torch.nn.ELU()


@dataclass
class SubgraphDataloaderConfig(DataloaderConfig):
    input_nodes: Optional[Union[Tensor, Sequence[int]]] = None
    use_parallel_sampling: bool = False


@dataclass
class GNSTrainingConfig(TrainingConfig):
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    subgraph_loader: SubgraphDataloaderConfig = field(default_factory=SubgraphDataloaderConfig)
