# --- batchers used by gns.py ---
from typing import Optional, Union, Sequence
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
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

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            seed_mask=seed_mask,
            subset=subset,
        ).to(self.device)

        return data



class InputsInjector:
    """
    Prepares (sub)graphs for batched inference or training by injecting global input parameters
    into node features and replicating the graph structure.

    Args:
        device (Union[str, torch.device]): Target device where the batched graph will reside.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @cr('InputsInjector.replicate_inject')
    def replicate_inject(
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
        N = getattr(graph, 'num_nodes', None)
        x = getattr(graph, 'x', None)
        edge_attr = getattr(graph, 'edge_attr', None)
        edge_index = getattr(graph, 'edge_index', None)
        seed_mask = getattr(graph, 'seed_mask', None)
        subset = getattr(graph, 'subset', None)

        assert N is not None, "Graph must have 'num_nodes' attribute."
        assert x is not None, "Graph must have 'x' attribute."
        assert edge_index is not None, "Graph must have 'edge_index' attribute."
        assert edge_attr is not None, "Graph must have 'edge_attr' attribute."
        if seed_mask is not None:
            assert subset is not None, "Graph must have 'subset' attribute if 'seed_mask' exists."
        else:
            assert getattr(graph, 'subset', None) is None, "Graph has ambiguous 'subset' attribute without 'seed_mask'."



        # 1. Node features with injected global inputs
        x_repeated = x.repeat(B, 1)                                           # [B*N, F]
        inputs_repeated = inputs_batch.repeat_interleave(N, dim=0)            # [B*N, D]
        x_repeated_injected = torch.cat([inputs_repeated, x_repeated], dim=1) # [B*N, F+D]

        # 2. Edge features
        edge_attr_repeated = edge_attr.repeat(B, 1)                   # [B*E, A]

        # 3. Edge index with offsets
        edge_index_expanded = edge_index.unsqueeze(0).expand(B, -1, -1)       # [B, 2, E]
        node_offset = torch.arange(B, device=graph.edge_index.device).view(B, 1, 1) * N
        edge_index_batch = edge_index_expanded + node_offset
        edge_index_batch = edge_index_batch.permute(1, 0, 2).reshape(2, -1)         # [2, B*E]

        # 4. Seed mask
        all_nodes_are_seeds = torch.ones(N * B, dtype=torch.bool, device=self.device)
        seed_mask_repeated = seed_mask.repeat(B) if seed_mask is not None else all_nodes_are_seeds

        # 5. Targets
        targets_flat = None
        if targets_batch is not None:
            if subset is not None:
                # Filter the full graph node labels to match the subgraph nodes.
                targets_batch_subgraph = targets_batch[:, subset, :]
            targets_flat = targets_batch_subgraph.reshape(-1, targets_batch.shape[-1])  # [B*N, O]

        data_kwargs = dict(
            x=x_repeated_injected,
            edge_index=edge_index_batch,
            edge_attr=edge_attr_repeated,
            seed_mask=seed_mask_repeated,
        )

        if targets_flat is not None:
            data_kwargs["y"] = targets_flat

        return Data(**data_kwargs).to(self.device)


class _ShapeValidator:
    """
    Input validator for the GNS model.

    This class encapsulates input validation logic for GNS, ensuring that inputs to
    `predict`, `fit`, or other routines conform to the expected shapes and dimensions.

    Supports both raw tensor inputs and datasets containing batched inputs and targets.

    Args:
        input_dim (int): Expected dimensionality of each input vector (D).
        output_dim (int, optional): Expected number of output features (F) per node. Used for target validation.
    """

    def __init__(self, input_dim: int, output_dim: int, num_nodes: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes

    def validate(self, X: Union[Tensor, TorchDataset]) -> None:
        """
        Validate the input type and shape.

        Args:
            X (Tensor or TorchDataset): Input to validate. Either:
                - Tensor of shape [D]
                - Dataset yielding (x, y) pairs with shapes:
                    - x: [D]
                    - y: [N, F] if `output_dim` is specified

        Raises:
            ValueError or TypeError if the input is malformed.
        """
        if isinstance(X, Tensor):
            self._validate_tensor(X)
        elif isinstance(X, TorchDataset):
            self._validate_dataset(X)
        else:
            raise ValueError(f"Invalid dataset of type {type(X)} for {self.__class__.__name__}: {e}") from e

    def _validate_tensor(self, x: Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected input Tensor of shape [B, D], got {x.shape}")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input feature dimension mismatch: expected {self.input_dim}, got {x.shape[1]}")

    def _validate_dataset(self, dataset: TorchDataset) -> None:
        try:
            sample = dataset[0]
        except Exception as e:
            raise ValueError("Failed to access first sample of the dataset for validation.") from e
        if isinstance(sample, (tuple, list)):
            x_sample, y_sample = sample
        else:
            x_sample, y_sample = sample, None

        if x_sample.ndim != 1:
            raise ValueError(f"Expected input sample of shape [D], got {x_sample.shape}")
        if x_sample.shape[0] != self.input_dim:
            raise ValueError(f"Input sample dimension mismatch: expected {self.input_dim}, got {x_sample.shape[0]}")

        if y_sample is not None:
            if y_sample.ndim != 2:
                raise ValueError(f"Expected target sample of shape [N, F], got {y_sample.shape}")
            if y_sample.shape[0] != self.num_nodes:
                raise ValueError(f"Target sample node count mismatch: expected {self.num_nodes}, got {y_sample.shape[0]}")
            if y_sample.shape[1] != self.output_dim:
                raise ValueError(f"Target output dim mismatch: expected {self.output_dim}, got {y_sample.shape[-1]}")


class _GNSHelpers:
    def __init__(self, device: torch.device, graph: Graph, num_msg_passing_layers: int):
        self.device = device
        self.graph = graph
        self.num_msg_passing_layers = num_msg_passing_layers

    def init_dataloader(self, X: Union[Tensor, TorchDataset], **kwargs) -> DataLoader:
        """
        Initialize a DataLoader from a Tensor or Dataset.

        Args:
            X (Tensor or Dataset): Input data.
            batch_size (int): Batch size (defaults to full batch if X is a Tensor).
            shuffle (bool): Whether to shuffle the data (only for Dataset).
            num_workers (int): Dataloader workers.
            pin_memory (bool): Pin memory for CUDA acceleration.

        Returns:
            DataLoader: Configured dataloader.
        """
        default_pin = self.device.type == "cuda" and torch.cuda.is_available()
        batch_size = kwargs.get("batch_size", 15)
        shuffle = kwargs.get("shuffle", True)
        num_workers = kwargs.get("num_workers", 0)
        pin_memory = kwargs.get("pin_memory", default_pin)

        if isinstance(X, Tensor):
            dataset = TensorDataset(X.cpu())
            return DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        elif isinstance(X, (TorchDataset)):
            return DataLoader(
                X,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        else:
            raise TypeError(f"Unsupported input type: {type(X)}")


    def init_subgraph_loader(self, batch_size: int = 256, input_nodes=None) -> ManualNeighborLoader:
        return ManualNeighborLoader(
            device=self.device,
            base_graph=self.graph,
            num_hops=self.num_msg_passing_layers,
            input_nodes=input_nodes,
            batch_size=batch_size,
            shuffle=True,
        )