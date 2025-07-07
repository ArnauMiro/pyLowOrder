# --- batchers used by gns.py ---
from typing import Optional, Union, Sequence
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from . import Graph
from ... import cr

class ManualNeighborLoader:
    """
    Lightweight replacement for PyG's NeighborLoader without requiring torch_sparse or pyg-lib.

    Performs k-hop sampling over a subset of seed nodes and yields mini-batches of subgraphs.

    Args:
        device (torch.device): Target device for output subgraphs.
        base_graph (Graph): Input graph with:
            - node_features: Tensor [N, F]
            - edge_index: LongTensor [2, E]
            - edge_features: Tensor [E, A]
        num_hops (int): Number of message-passing layers (L).
        batch_size (int): Number of seed nodes per subgraph batch (S).
        input_nodes (Union[Tensor, Sequence[int]], optional): Nodes to sample from.
            Can be:
                - LongTensor [K] of indices
                - BoolTensor [N] as mask
                - list[int]
            If None, all nodes are used. Default: None.
        shuffle (bool): Whether to shuffle input_nodes. Default: True.

    Yields:
        Data: Subgraph with:
            - node_features: Tensor [N', F]
            - edge_index: LongTensor [2, E']
            - edge_features: Tensor [E', A]
            - seed_mask: BoolTensor [N'], where True identifies seed nodes
    """

    def __init__(
        self,
        device: torch.device,
        base_graph: Data,
        num_hops: int,
        batch_size: int = 256,
        input_nodes: Optional[Union[Tensor, Sequence[int]]] = None,
        shuffle: bool = True,
    ) -> None:
        self.base_graph = base_graph
        self.num_hops = num_hops
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        if input_nodes is None:
            self.input_nodes = torch.arange(base_graph.num_nodes, device=self.device)
        else:
            input_nodes = torch.as_tensor(input_nodes, device=self.device)

            if input_nodes.dtype == torch.bool:
                if input_nodes.ndim != 1 or input_nodes.size(0) != base_graph.num_nodes:
                    raise ValueError("Boolean mask must have shape [N]")
                self.input_nodes = input_nodes.nonzero(as_tuple=False).view(-1)

            elif input_nodes.dtype in (torch.int32, torch.int64):
                self.input_nodes = input_nodes

            else:
                raise TypeError("input_nodes must be LongTensor, BoolTensor, or list of ints.")

    def __iter__(self):
        indices = self.input_nodes
        if self.shuffle:
            indices = indices[torch.randperm(indices.size(0), device=self.device)]

        for i in range(0, indices.size(0), self.batch_size):
            yield self.sample(indices[i:i + self.batch_size])

    def __len__(self) -> int:
        """Number of subgraph batches per epoch."""
        return (self.input_nodes.size(0) + self.batch_size - 1) // self.batch_size

    @cr('ManualNeighborLoader.sample')
    def sample(self, seed_nodes: Tensor) -> Data:
        """
        Extracts a subgraph around the given seed nodes using k-hop sampling.

        Args:
            seed_nodes (Tensor): LongTensor [S] of seed node indices.

        Returns:
            Data: Subgraph with:
                - node_features: Tensor [N', F]
                - edge_index: LongTensor [2, E']
                - edge_features: Tensor [E', A]
                - seed_mask: BoolTensor [N'], where True identifies seed nodes
        """
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            seed_nodes,
            num_hops=self.num_hops,
            edge_index=self.base_graph.edge_index,
            relabel_nodes=True
        )

        node_features = self.base_graph.node_features[subset]
        edge_features = self.base_graph.edge_features[edge_mask]

        seed_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=self.device)
        seed_mask[mapping] = True

        return Data(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            seed_mask=seed_mask
        ).to(self.device)



class GraphPreparer:
    """
    Prepares (sub)graphs for batched inference or training by injecting global input parameters
    into node features and replicating the graph structure.

    Args:
        device (Union[str, torch.device]): Target device where the batched graph will reside.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @cr('GraphPreparer.prepare_batch')
    def prepare_batch(
        self,
        graph: Data,
        inputs_batch: Tensor,
        targets_batch: Optional[Tensor] = None,
    ) -> Data:
        """
        Prepares a batched graph from a single (sub)graph by replicating it across input parameters
        and injecting those into the node features.

        Args:
            graph (Data): A graph with `node_features`, `edge_index`, `edge_features`, and `seed_mask`.
            inputs_batch (Tensor): Tensor of shape [B, D] representing B input conditions.
            targets_batch (Optional[Tensor]): Optional tensor of shape [B, N, O] with target node labels.

        Returns:
            Data: A single `torch_geometric.data.Data` object with:
                - node_features: shape [B * N, F + D]
                - edge_index: shape [2, B * E]
                - edge_features: shape [B * E, A]
                - seed_mask: shape [B * N]
                - node_labels (optional): shape [B * N, O]
        """
        B = inputs_batch.size(0)
        N = graph.num_nodes

        # 1. Node features with injected global inputs
        nf_repeated = graph.node_features.repeat(B, 1)                       # [B*N, F]
        inputs_repeated = inputs_batch.repeat_interleave(N, dim=0)          # [B*N, D]
        all_node_features = torch.cat([inputs_repeated, nf_repeated], dim=1)# [B*N, F+D]

        # 2. Edge features
        edge_features_repeated = graph.edge_features.repeat(B, 1)           # [B*E, A]

        # 3. Edge index with offsets
        edge_index_expanded = graph.edge_index.unsqueeze(0).expand(B, -1, -1)       # [B, 2, E]
        node_offset = torch.arange(B, device=graph.edge_index.device).view(B, 1, 1) * N
        edge_index_batch = edge_index_expanded + node_offset
        edge_index_batch = edge_index_batch.permute(1, 0, 2).reshape(2, -1)          # [2, B*E]

        # 4. Seed mask
        seed_mask = graph.seed_mask.repeat(B)                                # [B*N]

        # 5. Targets
        targets_flat = None
        if targets_batch is not None:
            targets_flat = targets_batch.reshape(-1, targets_batch.shape[-1])  # [B*N, O]

        data_kwargs = dict(
            node_features=all_node_features,
            edge_index=edge_index_batch,
            edge_features=edge_features_repeated,
            seed_mask=seed_mask,
        )

        if targets_flat is not None:
            data_kwargs["node_labels"] = targets_flat

        return Data(**data_kwargs).to(self.device)


class SubgraphBatcher:
    """
    Combines a subgraph sampler and a batch preparer into a single callable.
    
    Used to generate inference-ready batched graphs for GNN training and evaluation.

    Args:
        sampler (ManualNeighborLoader): Subgraph extractor.
        preparer (GraphPreparer): Batch preparer for injecting inputs and replicating structure.
    """
    def __init__(self, sampler: ManualNeighborLoader, preparer: GraphPreparer) -> None:
        self.sampler = sampler
        self.preparer = preparer

    @cr('SubgraphBatcher.__call__')
    def __call__(
        self,
        inputs_batch: Tensor,
        targets_batch: Optional[Tensor] = None,
        seed_nodes: Optional[Tensor] = None,
    ) -> Data:
        """
        Prepares a batched subgraph from input parameters and seed nodes.

        Args:
            inputs_batch (Tensor): Shape [B, D], global input conditions.
            targets_batch (Tensor, optional): Shape [B, N, O] with target node labels.
            seed_nodes (Tensor, optional): Seed node indices for subgraph extraction.

        Returns:
            Data: Batched PyG graph ready for model input.
        """
        if seed_nodes is not None:
            subgraph = self.sampler.sample(seed_nodes)
        else:
            # Full-graph inference mode: treat all nodes as seed nodes
            subgraph = self.sampler.base_graph
            subgraph.seed_mask = torch.ones(subgraph.num_nodes, dtype=torch.bool, device=self.preparer.device)

        return self.preparer.prepare_batch(subgraph, inputs_batch, targets_batch)

##################################################################################################################
##################################################################################################################
##################################################################################################################


# class ListBasedSubgraphBatcher:
#     def __init__(self, graph: Union[Graph, Data], num_hops: int, input_dim: int, device: torch.device):
#         self.graph = graph
#         self.num_hops = num_hops
#         self.input_dim = input_dim
#         self.device = device
#         self.preparer = GraphPreparer(input_dim, device)

#     @cr('ListBasedSubgraphBatcher.__call__')
#     def __call__(
#         self,
#         inputs_batch: Tensor,               # [B, D]
#         targets_batch: Tensor,             # [B, N, output_dim]
#         seed_nodes: Optional[Tensor] = None
#     ) -> Batch:
#         subset, edge_index, mapping, edge_mask = k_hop_subgraph(
#             seed_nodes, self.num_hops, self.graph.edge_index, relabel_nodes=True
#         )

#         nf_subg = self.graph.node_features[subset]
#         ef_subg = self.graph.edge_features[edge_mask]
#         B = inputs_batch.size(0)
#         targets_batch_subg = targets_batch[:, subset, :]    # [B, N_subg, output_dim]

#         graphs = []
#         for i in range(B):
#             seed_mask = torch.zeros(nf_subg.size(0), dtype=torch.bool, device=self.device)
#             seed_mask[mapping] = True

#             data = Data(
#                 node_features=nf_subg,
#                 edge_index=edge_index,
#                 edge_features=ef_subg
#             )
#             graph = self.preparer(data, inputs_batch[i], seed_mask=seed_mask, node_labels=targets_batch_subg[i])
#             graphs.append(graph)

#         return Batch.from_data_list(graphs).to(self.device)
