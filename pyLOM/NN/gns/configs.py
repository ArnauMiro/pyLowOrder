from dataclasses import dataclass
from typing import Union, Optional, Any
import torch

@dataclass
class GNSConfig:
    graph_path: str
    input_dim: int
    latent_dim: int
    output_dim: int
    hidden_size: int
    num_msg_passing_layers: int
    encoder_hidden_layers: int
    decoder_hidden_layers: int
    message_hidden_layers: int
    update_hidden_layers: int
    p_dropouts: float = 0.0
    activation: Union[str, torch.nn.Module] = torch.nn.ELU()
    seed: Optional[int] = None
    device: Union[str, torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    epochs: int = 100
    lr: float = 1e-4
    lr_gamma: float = 0.1
    lr_scheduler_step: int = 1
    loss_fn: Any = torch.nn.MSELoss(reduction='mean')
    optimizer: Any = torch.optim.Adam
    scheduler: Any = torch.optim.lr_scheduler.StepLR
    batch_size: int = 1
    node_batch_size: int = 256
    num_workers: int = 1
    input_nodes: Optional[torch.Tensor] = None
    verbose: Union[bool, int] = 1  # False = no logging, True = every epoch, or int = log every N epochs
