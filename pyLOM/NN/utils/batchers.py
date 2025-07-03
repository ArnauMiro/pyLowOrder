# --- batchers used by gns.py ---
from typing import Optional, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph

from . import Graph
from ... import cr


class GraphPreparer:
    """
    Prepares a graph or subgraph for inference by injecting operational input parameters
    into node features and replicating the graph structure over the batch dimension.

    This class is used to generate inference-ready graphs from a base graph
    and a batch of operational inputs.

    Args:
        input_dim (int): Dimensionality of the operational parameters.
        device (Union[str, torch.device]): Target device for computation.
    """

    def __init__(self, input_dim: int, device: Union[str, torch.device]) -> None:
        self.input_dim = input_dim
        self.device = torch.device(device)

    @cr('GraphPreparer.__call__')
    def __call__(
        self,
        inputs: Tensor,
        graph: Data,
        seed_mask: Optional[Tensor] = None,
        node_labels: Optional[Tensor] = None
    ) -> Data:
        """
        Prepare a batched graph for inference or training using a base graph and a batch of operational parameters.

        Args:
            inputs (Tensor): A tensor of shape [B, D] representing B sets of operational parameters.
            graph (Data): A full graph with node_features, edge_index, edge_features.
            seed_mask (Tensor, optional): Boolean mask of seed nodes in the graph. Default is None.
            targets (Tensor, optional): Optional tensor of shape [B * N, output_dim] representing node targets.

        Returns:
            Data: A new graph containing B replicated graphs with injected input parameters.
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # [1, D]

        B = inputs.size(0)
        N = graph.node_features.size(0)

        nf_repeated = graph.node_features.repeat(B, 1)               # [B*N, F]
        inputs_repeated = inputs.repeat_interleave(N, dim=0)         # [B*N, D]
        all_node_features = torch.cat([inputs_repeated, nf_repeated], dim=1) # [B*N, F+D]

        edge_index = torch.cat([
            graph.edge_index + i * N for i in range(B)
        ], dim=1)                                                    # [2, B*E]
        edge_features = graph.edge_features.repeat(B, 1)             # [B*E, A]

        data_kwargs = dict(
            node_features=all_node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )
        if seed_mask is not None:
            data_kwargs["seed_mask"] = seed_mask
        if node_labels is not None:
            data_kwargs["node_labels"] = node_labels

        return Data(**data_kwargs).to(self.device)


class SubgraphBatcher:
    def __init__(self, graph: Union[Graph, Data], num_hops: int, input_dim: int, device: torch.device) -> None:
        self.graph = graph
        self.num_hops = num_hops
        self.input_dim = input_dim
        self.device = device
        self.preparer = GraphPreparer(input_dim, device)

    @cr('SubgraphBatcher.__call__')
    def __call__(
        self,
        inputs_batch: Tensor,             # [B, D]
        targets_batch: Tensor,            # [B, N, output_dim]
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

        nf_subg = self.graph.node_features[subset]               # [N_subg, F]
        ef_subg = self.graph.edge_features[edge_mask]            # [E_subg, A]
        targets_batch_subg = targets_batch[:, subset, :]         # [B, N_subg, output_dim]

        B = inputs_batch.size(0)
        N_subg = nf_subg.size(0)
        seed_mask = torch.zeros(B * N_subg, dtype=torch.bool, device=self.device)
        offsets = torch.arange(B, device=self.device).unsqueeze(1) * N_subg
        indices = offsets + mapping.unsqueeze(0)  # [B, num_seed_nodes]
        seed_mask[indices.view(-1)] = True


        targets_flat = targets_batch_subg.reshape(-1, targets_batch_subg.shape[-1])            # [B*N_subg, output_dim]

        subgraph = Data(
            node_features=nf_subg,
            edge_index=edge_index,
            edge_features=ef_subg
        )
        return self.preparer(subgraph, inputs_batch, seed_mask=seed_mask, node_labels=targets_flat)


class ListBasedSubgraphBatcher:
    def __init__(self, graph: Union[Graph, Data], num_hops: int, input_dim: int, device: torch.device):
        self.graph = graph
        self.num_hops = num_hops
        self.input_dim = input_dim
        self.device = device
        self.preparer = GraphPreparer(input_dim, device)

    @cr('ListBasedSubgraphBatcher.__call__')
    def __call__(
        self,
        inputs_batch: Tensor,               # [B, D]
        targets_batch: Tensor,             # [B, N, output_dim]
        seed_nodes: Optional[Tensor] = None
    ) -> Batch:
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            seed_nodes, self.num_hops, self.graph.edge_index, relabel_nodes=True
        )

        nf_subg = self.graph.node_features[subset]
        ef_subg = self.graph.edge_features[edge_mask]
        B = inputs_batch.size(0)
        targets_batch_subg = targets_batch[:, subset, :]    # [B, N_subg, output_dim]

        graphs = []
        for i in range(B):
            seed_mask = torch.zeros(nf_subg.size(0), dtype=torch.bool, device=self.device)
            seed_mask[mapping] = True

            data = Data(
                node_features=nf_subg,
                edge_index=edge_index,
                edge_features=ef_subg
            )
            graph = self.preparer(data, inputs_batch[i], seed_mask=seed_mask, node_labels=targets_batch_subg[i])
            graphs.append(graph)

        return Batch.from_data_list(graphs).to(self.device)
