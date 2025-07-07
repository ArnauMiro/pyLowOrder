# --- batchers used by gns.py ---
from typing import Optional, Union, Sequence
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from .. import Graph
from ... import cr

class ManualNeighborLoader:
    """
    Lightweight replacement for PyG's NeighborLoader without requiring torch_sparse or pyg-lib.

    Performs k-hop sampling over a subset of seed nodes and yields mini-batches of subgraphs.

    Args:
        device (torch.device): Target device for output subgraphs.
        base_graph (Graph): Input graph with:
            - x: Tensor [N, F]
            - edge_index: LongTensor [2, E]
            - edge_attr: Tensor [E, A]
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
            - x: Tensor [N', F]
            - edge_index: LongTensor [2, E']
            - edge_attr: Tensor [E', A]
            - seed_mask: BoolTensor [N'], where True identifies seed nodes
    """

    def __init__(
        self,
        device: torch.device,
        base_graph: Graph,
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
                - x: Tensor [N', F]
                - edge_index: LongTensor [2, E']
                - edge_attr: Tensor [E', A]
                - seed_mask: BoolTensor [N'], where True identifies seed nodes
        """
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            seed_nodes,
            num_hops=self.num_hops,
            edge_index=self.base_graph.edge_index,
            relabel_nodes=True
        )

        x = self.base_graph.x[subset]
        edge_attr = self.base_graph.edge_attr[edge_mask]

        seed_mask = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
        seed_mask[mapping] = True

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            seed_mask=seed_mask
        ).to(self.device)



class InputsInjector:
    """
    Prepares (sub)graphs for batched inference or training by injecting global input parameters
    into node features and replicating the graph structure.

    Args:
        device (Union[str, torch.device]): Target device where the batched graph will reside.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @cr('InputsInjector.inject_replicate')
    def inject_replicate(
        self,
        graph: Data,
        inputs_batch: Tensor,
        targets_batch: Optional[Tensor] = None,
    ) -> Data:
        """
        Prepares a batched graph from a single (sub)graph by replicating it across input parameters
        and injecting those into the node features.

        Args:
            graph (Data): A graph with `x`, `edge_index`, `edge_attr`, and `seed_mask`.
            inputs_batch (Tensor): Tensor of shape [B, D] representing B input conditions.
            targets_batch (Optional[Tensor]): Optional tensor of shape [B, N, O] with target node labels.

        Returns:
            Data: A single `torch_geometric.data.Data` object with:
                - x: shape [B * N, F + D]
                - edge_index: shape [2, B * E]
                - edge_attr: shape [B * E, A]
                - seed_mask: shape [B * N]
                - y (optional): shape [B * N, O]
        """
        B = inputs_batch.size(0)
        N = graph.num_nodes

        # 1. Node features with injected global inputs
        nf_repeated = graph.x.repeat(B, 1)                       # [B*N, F]
        inputs_repeated = inputs_batch.repeat_interleave(N, dim=0)          # [B*N, D]
        all_x = torch.cat([inputs_repeated, nf_repeated], dim=1)# [B*N, F+D]

        # 2. Edge features
        edge_attr_repeated = graph.edge_attr.repeat(B, 1)           # [B*E, A]

        # 3. Edge index with offsets
        edge_index_expanded = graph.edge_index.unsqueeze(0).expand(B, -1, -1)       # [B, 2, E]
        node_offset = torch.arange(B, device=graph.edge_index.device).view(B, 1, 1) * N
        edge_index_batch = edge_index_expanded + node_offset
        edge_index_batch = edge_index_batch.permute(1, 0, 2).reshape(2, -1)          # [2, B*E]

        # 4. Seed mask
        seed_mask = getattr(graph, 'seed_mask', None)
        if seed_mask is not None:
            # Repeat seed mask for each batch
            seed_mask = seed_mask.repeat(B)
        else:
            # If no seed mask exists, create a default one
            # This assumes all nodes are seeds in the full graph inference mode
            seed_mask = torch.ones(N * B, dtype=torch.bool, device=self.device)

        # 5. Targets
        targets_flat = None
        if targets_batch is not None:
            targets_flat = targets_batch.reshape(-1, targets_batch.shape[-1])  # [B*N, O]

        data_kwargs = dict(
            x_injected=all_x,
            edge_index=edge_index_batch,
            edge_attr=edge_attr_repeated,
            seed_mask=seed_mask,
        )

        if targets_flat is not None:
            data_kwargs["y"] = targets_flat

        return Data(**data_kwargs).to(self.device)
