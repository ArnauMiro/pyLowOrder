import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import ELU
from torch_geometric.nn import MessagePassing

from ..utils.torch_utils import snap, assert_finite, ensure_2d_f32_contig, safe_linear
from ... import cr

class GNSMLP(nn.Module):
    r"""
    Simple feedforward MLP with activation and dropout, designed for internal use in GNS.

    Args:
        input_size (int): Input size.
        output_size (int): Output size.
        hidden_size (int): Number of units in each hidden layer.
        num_hidden_layers (int): Number of hidden layers.
        drop_p (float): Dropout probability.
        activation (nn.Module): Activation function.
    """
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers=1, drop_p=0.5, activation=ELU()):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.activation = activation
        self.dropout = nn.Dropout(p=drop_p)

    @cr('GNSMLP.forward')
    def forward(self, x):
        # print("GNSMLP forward")
        # print("dropout module:", self.dropout, "training:", self.training)
        assert not getattr(self.dropout, 'inplace', False)

        for i, layer in enumerate(self.layers[:-1]):
            # print("Layer:", layer)
            snap(x, f"[mp] before Linear{i}")
            x = ensure_2d_f32_contig(x)
            assert_finite(x, f"[mp] Linear{i} input")

            # Ultra-safe path: float64 + chunk=64 + single-thread
            x = safe_linear(x, layer, chunk=64, compute_dtype=torch.float64,
                            single_thread_mm=True, debug_name=f"L{i}")
            x = self.activation(x)

            # snap(x, f"[mp] after act{i}")
            x = self.dropout(x)
            # snap(x, f"[mp] after dropout{i}")

        # Final (no activation)
        # snap(x, "[mp] before final Linear")
        x = ensure_2d_f32_contig(x)
        assert_finite(x, "[mp] final Linear input")
        x = safe_linear(x, self.layers[-1], chunk=64, compute_dtype=torch.float64,
                        single_thread_mm=True, debug_name="Lfinal")
        # snap(x, "[mp] final out")
        return x

        # return self.layers[-1](x)



class MessagePassingLayer(MessagePassing):
    r"""
    Message passing layer using mean aggregation and MLPs for message and update functions.

    Args:
        in_channels (int): Input feature size for messages.
        out_channels (int): Output feature size after message passing.
        hidden_size (int): Hidden size for internal MLPs.
        message_hidden_layers (int): Number of hidden layers in the message MLP.
        update_hidden_layers (int): Number of hidden layers in the update MLP.
        activation (nn.Module): Activation function (e.g., nn.ELU()).
        drop_p (float): Dropout probability.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        message_hidden_layers: int = 1,
        update_hidden_layers: int = 1,
        activation: nn.Module = ELU(),
        drop_p: float = 0.0,
    ):
        super().__init__(aggr='mean')
        self.dropout = nn.Dropout(p=drop_p)

        self.phi = GNSMLP(
            input_size=in_channels,
            output_size=out_channels,
            hidden_size=hidden_size,
            num_hidden_layers=message_hidden_layers,
            drop_p=drop_p,
            activation=activation,
        )

        self.gamma = GNSMLP(
            input_size=2 * out_channels,
            output_size=out_channels,
            hidden_size=hidden_size,
            num_hidden_layers=update_hidden_layers,
            drop_p=drop_p,
            activation=activation,
        )

    @cr('MessagePassingLayer.forward')
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    @cr('MessagePassingLayer.message')
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.phi(torch.cat([x_i, x_j, edge_attr], dim=1))

    @cr('MessagePassingLayer.update')
    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return self.gamma(torch.cat([x, aggr_out], dim=1))