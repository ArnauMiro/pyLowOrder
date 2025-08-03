#!/usr/bin/env python
#
# pyLOM, NN
#
# Dataclass configurations for NN models.
#
# Last rev: 28/07/2025

from dataclasses import dataclass
from typing import Optional, Union, Any
import torch


@dataclass
class GNSModelParams:
    """
    Hyperparameters for a GNS model.

    All values must be pre-resolved before instantiating this object.
    In particular, `activation` must be a torch.nn.Module instance, and
    `device` must be a torch.device.

    Fields:
        input_dim (int): Number of external input features.
        latent_dim (int): Latent dimension used throughout the network.
        output_dim (int): Number of output features.
        hidden_size (int): Width of the MLP layers.
        num_msg_passing_layers (int): Number of message-passing steps (depth of GNN).
        encoder_hidden_layers (int): Number of hidden layers in the encoder MLP.
        decoder_hidden_layers (int): Number of hidden layers in the decoder MLP.
        message_hidden_layers (int): Number of hidden layers in the message MLP.
        update_hidden_layers (int): Number of hidden layers in the update MLP.
        groupnorm_groups (int): Number of groups for GroupNorm.
        p_dropouts (float): Dropout probability used across the model.
        activation (torch.nn.Module): Activation function instance (e.g. nn.ELU()).
        seed (int, optional): Random seed for reproducibility.
        device (torch.device): Device to use for training and inference.
    """
    input_dim: int
    latent_dim: int
    output_dim: int
    hidden_size: int
    num_msg_passing_layers: int
    encoder_hidden_layers: int
    decoder_hidden_layers: int
    message_hidden_layers: int
    update_hidden_layers: int
    groupnorm_groups: int = 2
    p_dropouts: float = 0.0
    activation: torch.nn.Module = torch.nn.ELU()
    seed: Optional[int] = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GNSTrainingParams:
    """
    Training configuration for a GNS model.

    Fields:
        loss_fn (torch.nn.Module): Loss function instance (e.g. nn.MSELoss()).
        optimizer (type): Optimizer class (e.g. torch.optim.Adam).
        scheduler (type): LR scheduler class (e.g. torch.optim.lr_scheduler.StepLR).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for external inputs.
        node_batch_size (int): Number of seed nodes per subgraph batch.
        lr (float): Learning rate.
        lr_gamma (float): Gamma factor for LR scheduler.
        lr_scheduler_step (int): Step interval for LR scheduler.
        num_workers (int): Number of workers for dataloaders.
        pin_memory (bool): Whether to pin memory for dataloaders.
        input_nodes (Tensor, optional): Seed nodes for subgraph sampling (if any).
        verbose (bool or int): Logging verbosity.
    """
    loss_fn: Any = torch.nn.MSELoss(reduction='mean')
    optimizer: Any = torch.optim.Adam
    scheduler: Any = torch.optim.lr_scheduler.StepLR
    epochs: int = 100
    batch_size: int = 32
    node_batch_size: int = 256
    lr: float = 1e-4
    lr_gamma: float = 0.1
    lr_scheduler_step: int = 1
    num_workers: int = 1
    pin_memory: bool = True
    input_nodes: Optional[torch.Tensor] = None
    verbose: Union[bool, int] = 1  # False = no logging, True = every epoch, or int = log every N epochs
