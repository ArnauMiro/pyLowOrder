from typing import Optional, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph

from . import Graph

class VectorizedBatcher:
    def __init__(self, graph: Union[Graph, Data], num_hops: int, input_dim: int, device: torch.device):
        self.graph = graph
        self.num_hops = num_hops
        self.input_dim = input_dim
        self.device = device

    def __call__(
        self,
        op_params: Tensor,             # [B, D]
        y_batch: Tensor,               # [B, N, output_dim]
        seed_nodes: Optional[Tensor] = None
    ) -> Data:
        if seed_nodes is not None:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                seed_nodes, self.num_hops, self.graph.edge_index, relabel_nodes=True
            )
        else:
            subset = torch.arange(self.graph.num_nodes, device=self.device)
            edge_index = self.graph.edge_index
            edge_mask = slice(None)
            mapping = torch.arange(self.graph.num_nodes, device=self.device)

        # Subgraph data
        x_subg = self.graph.x[subset]                            # [N_sub, F]
        edge_attr_subg = self.graph.edge_attr[edge_mask]         # [E_sub, A]
        y_batch_subg = y_batch[:, subset, :]                     # [B, N_sub, output_dim]

        B = op_params.size(0)
        N_sub = x_subg.size(0)

        # Expand node features
        x_repeated = x_subg.repeat(B, 1)                         # [B * N_sub, F]
        op_repeated = op_params.repeat_interleave(N_sub, dim=0) # [B * N_sub, D]
        x_repeated[:, :self.input_dim] = op_repeated            # in-place overwrite

        # Expand edges
        edge_index_repeated = torch.cat([
            edge_index + i * N_sub for i in range(B)
        ], dim=1)                                                # [2, B * E_sub]
        edge_attr_repeated = edge_attr_subg.repeat(B, 1)         # [B * E_sub, A]

        # Expand targets
        y_expanded = y_batch_subg.reshape(-1, y_batch.size(-1))  # [B * N_sub, output_dim]

        # Create seed mask
        seed_mask = torch.zeros(B * N_sub, dtype=torch.bool, device=self.device)
        for i in range(B):
            seed_mask[i * N_sub + mapping] = True

        return Data(
            x=x_repeated,
            edge_index=edge_index_repeated,
            edge_attr=edge_attr_repeated,
            y=y_expanded,
            seed_nodes=seed_mask,
        ).to(self.device)


class ListBasedBatcher:
    def __init__(self, graph: Union[Graph, Data], num_hops: int, input_dim: int, device: torch.device):
        self.graph = graph
        self.num_hops = num_hops
        self.input_dim = input_dim
        self.device = device

    def __call__(
        self,
        op_params: Tensor,               # [B, D]
        y_batch: Tensor,                 # [B, N, output_dim]
        seed_nodes: Tensor               # [S]
    ) -> Batch:
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            seed_nodes, self.num_hops, self.graph.edge_index, relabel_nodes=True
        )
        x_subg = self.graph.x[subset]
        edge_attr_subg = self.graph.edge_attr[edge_mask]
        B, _, output_dim = y_batch.shape
        y_batch_subg = y_batch[:, subset, :]

        G_list = []
        for i in range(B):
            x = x_subg.clone()
            x[:, :self.input_dim] = op_params[i]
            y = y_batch_subg[i]  # [N_sub, output_dim]
            seed_mask = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
            seed_mask[mapping] = True

            data = Data(
                x=x, y=y, seed_nodes=seed_mask,
                edge_index=edge_index, edge_attr=edge_attr_subg
            )
            G_list.append(data)

        return Batch.from_data_list(G_list).to(self.device)
