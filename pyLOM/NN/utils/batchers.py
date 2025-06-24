from typing import Optional

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch

class VectorizedBatcher:
    def __init__(self, graph, num_hops, input_dim, device):
        self.graph = graph
        self.num_hops = num_hops
        self.input_dim = input_dim
        self.device = device

    def __call__(self, op_params: Tensor, y_batch: Tensor, seed_nodes: Optional[Tensor] = None) -> Data:
        if seed_nodes is not None:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                seed_nodes, self.num_hops, self.graph.edge_index, relabel_nodes=True
            )
        else:
            subset = torch.arange(self.graph.num_nodes, device=self.device)
            edge_index = self.graph.edge_index
            edge_attr = self.graph.edge_attr
            mapping = torch.arange(self.graph.num_nodes, device=self.device)
            edge_mask = slice(None)

        x_subg = self.graph.x[subset] # [N_sub, ...]
        edge_attr_subg = edge_attr[edge_mask]
        y_batch_subg = y_batch[:, subset]  # [B, N_sub]

        B = op_params.size(0)
        N_sub = x_subg.size(0)

        x_repeated = x_subg.repeat(B, 1)
        op_repeated = op_params.repeat_interleave(N_sub, dim=0)
        x_repeated[:, :self.input_dim] = op_repeated

        edge_index_repeated = torch.cat([
            edge_index + i * N_sub for i in range(B)
        ], dim=1)
        edge_attr_repeated = edge_attr_subg.repeat(B, 1)
        y_expanded = y_batch_subg.reshape(-1, 1)

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