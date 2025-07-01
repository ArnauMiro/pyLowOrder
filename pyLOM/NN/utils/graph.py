#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines.
#
# Last rev: 02/10/2024

# Built-in modules
import os
from typing import Optional, Tuple, Union, Dict
import warnings

# Third-party libraries
import numpy as np
import torch
from torch_geometric.data import Data

# Local modules
from .. import DEVICE
from ... import io, cr
from ...mesh import Mesh
from ...vmmath.geometric import edge_to_cells, wall_normals






class Graph(Data):
    def __init__(
        self,
        edge_index: torch.Tensor,
        node_features_dict: Dict[str, torch.Tensor],
        edge_features_dict: Dict[str, torch.Tensor],
        device: Union[str, torch.device] = None,
        # **custom_attr_dict: Any # Not yet implemented.
    ):
        r'''
        Initialize the Graph object. Node and edge attributes are stacked separately along dimension 1 for use in GNNs.

        Args:
            edge_index (torch.Tensor): Edge connectivity in COO format [2, num_edges].
            node_features_dict (Dict[str, torch.Tensor]): Dict of node features with shape [num_nodes, feature_dim] per entry.
            edge_features_dict (Dict[str, torch.Tensor]): Dict of edge features with shape [num_edges, feature_dim] per entry.
            device (Union[str, torch.device], optional): Computation device. Defaults to global DEVICE.
        '''
        if device is None:
            device = torch.device(DEVICE)  # Default to global DEVICE constant
        elif isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA not available. Falling back to CPU.")
            device = torch.device('cpu')

        # Ensure everything is on the correct device
        edge_index = edge_index.to(device)
        node_features_dict = {k: v.to(device) for k, v in node_features_dict.items()}
        edge_features_dict = {k: v.to(device) for k, v in edge_features_dict.items()}
        # custom_attr_dict = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in custom_attr_dict.items()} # Not yet implemented.

        # Concatenate node/edge attributes
        node_features = torch.cat(list(node_features_dict.values()), dim=1) if node_features_dict else None
        edge_features = torch.cat(list(edge_features_dict.values()), dim=1) if edge_features_dict else None

        # Build kwargs for Data
        data_kwargs = {
            'edge_index': edge_index,
            'node_features': node_features,
            'edge_features': edge_features,
            'num_nodes': next(iter(node_features_dict.values())).shape[0] if node_features_dict else None
            # **custom_attr_dict,
        }

        super().__init__(**data_kwargs)


        # Register individual attributes for user-friendly access
        for k, v in node_features_dict.items():
            setattr(self, k, v)
        for k, v in edge_features_dict.items():
            setattr(self, k, v)

        # store raw attribute dicts (for i/o operations)
        self.node_features_dict = node_features_dict
        self.edge_features_dict = edge_features_dict
        # self.custom_attr_dict = custom_attr_dict # Not yet implemented.

        self.validate()

    def validate(self) -> "Graph":
        """
        Validate the internal consistency of the graph structure and features.

        Returns:
            Graph: Self, for chaining.
        Raises:
            AssertionError: If any inconsistency is found.
        """
        self._validate_node_features()
        self._validate_edge_features()
        self._validate_edge_index()
        return self

    def _validate_node_features(self):
        assert isinstance(self.node_features_dict, dict), "node_features_dict must be a dictionary"
        assert len(set(self.node_features_dict)) == len(self.node_features_dict), "Duplicate keys in node_features_dict"
        
        # Check shapes
        for k, v in self.node_features_dict.items():
            assert isinstance(v, torch.Tensor), f"Node attribute '{k}' must be a tensor"
            assert v.shape[0] == self.num_nodes, f"Node attribute '{k}' has wrong number of nodes: expected {self.num_nodes}, got {v.shape[0]}"
            assert not torch.isnan(v).any(), f"Node attribute '{k}' contains NaNs"

        # Validate concatenation
        nf_cat = torch.cat(list(self.node_features_dict.values()), dim=-1)
        assert self.node_features.shape == nf_cat.shape, "node_features shape mismatch with concatenated node_features_dict"
        assert torch.allclose(self.node_features, nf_cat, atol=1e-6), "node_features do not match concatenated node_features_dict"

    def _validate_edge_features(self):
        assert isinstance(self.edge_features_dict, dict), "edge_features_dict must be a dictionary"
        assert len(set(self.edge_features_dict)) == len(self.edge_features_dict), "Duplicate keys in edge_features_dict"
        
        for k, v in self.edge_features_dict.items():
            assert isinstance(v, torch.Tensor), f"Edge attribute '{k}' must be a tensor"
            assert v.shape[0] == self.num_edges, f"Edge attribute '{k}' has wrong number of edges: expected {self.num_edges}, got {v.shape[0]}"
            assert not torch.isnan(v).any(), f"Edge attribute '{k}' contains NaNs"

        ef_cat = torch.cat(list(self.edge_features_dict.values()), dim=-1)
        assert self.edge_features.shape == ef_cat.shape, "edge_features shape mismatch with concatenated edge_features_dict"
        assert torch.allclose(self.edge_features, ef_cat, atol=1e-6), "edge_features do not match concatenated edge_features_dict"

    def _validate_edge_index(self):
        assert isinstance(self.edge_index, torch.Tensor), "edge_index must be a tensor"
        assert self.edge_index.shape[0] == 2, "edge_index must have shape [2, num_edges]"
        assert self.edge_index.shape[1] == self.num_edges, "edge_index second dimension must match number of edges"
        assert self.edge_index.max() < self.num_nodes, "edge_index contains invalid node indices"
        assert not torch.isnan(self.edge_index).any(), "edge_index contains NaNs"

    @cr('Graph.from_pyLOM_mesh')
    @classmethod
    def from_pyLOM_mesh(cls,
                        mesh: Mesh,
                        device: Optional[Union[str, torch.device]] = None,
                        # **custom_attr_dict: Dict[str, Any]  # Not yet implemented.
                        ) -> "Graph":
        r"""
        Create a Graph object from a pyLOM Mesh object. This method computes the node attributes and edge index/attributes from the mesh.
        Args:
            mesh (Mesh): A mesh object in pyLOM format.
            device (Optional[Union[str, torch.device]]): The device to use for the graph.
        Returns:
            Graph: A Graph object with the computed node attributes and edge index/attributes.
        """
        node_features_dict = cls._compute_node_features_dict(mesh)  # Get the node attributes
        edge_index, edge_features_dict = cls._compute_edge_index_and_attr_dict(mesh)  # Get the edge attributes

        graph = cls(
            edge_index=edge_index,
            node_features_dict = node_features_dict,
            edge_features_dict=edge_features_dict,
            device=device,
            # **custom_attr_dict # Not yet implemented.
            )

        return graph

    def save(self, fname: str, **kwargs):
        """
        Save the graph to disk. Supports .h5, .pt, .pkl.

        Args:
            fname (str): Output file path.
        """
        fmt = os.path.splitext(fname)[1][1:].lower()

        if fmt == 'h5':
            node_dict = self._to_pyLOM_format(self.node_features_dict)
            edge_dict = self._to_pyLOM_format(self.edge_features_dict)
            io.h5_save_graph_serial(
                fname,
                num_nodes=self.num_nodes,
                num_edges=self.num_edges,
                edge_index=self.edge_index.cpu().numpy(),
                nodeFeatrDict=node_dict,
                edgeFeatrDict=edge_dict,
                **kwargs
            )
        elif fmt in ['pt', 'pkl']:
            torch.save(self, fname)
        else:
            raise ValueError(f"Unsupported file format: {fmt}")

    @classmethod
    def load(cls, fname: str, **kwargs) -> "Graph":
        """
        Load a graph from disk. Supports .h5, .pt, .pkl.

        Args:
            fname (str): File path to load from.

        Returns:
            Graph: Loaded graph object.
        """
        fmt = os.path.splitext(fname)[1][1:].lower()

        if fmt == 'h5':
            num_nodes, num_edges, edge_index, nodeFeatrDict, edgeFeatrDict = io.h5_load_graph_serial(fname)
            init_kwargs = cls._from_pyLOM_format(edge_index, nodeFeatrDict, edgeFeatrDict)

            # Set device
            device = kwargs.get('device', DEVICE)
            init_kwargs['device'] = torch.device(device) if isinstance(device, str) else device

            # Construct graph
            graph = cls(**init_kwargs)

            # Validate structural consistency
            assert graph.num_nodes == num_nodes, f"Mismatch: loaded num_nodes={graph.num_nodes}, file={num_nodes}"
            assert graph.num_edges == num_edges, f"Mismatch: loaded num_edges={graph.num_edges}, file={num_edges}"
            return graph

        elif fmt in ['pt', 'pkl']:
            return torch.load(fname)

        else:
            raise ValueError(f"Unsupported file format: {fmt}")


    @staticmethod
    def _to_pyLOM_format(attr_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Union[int, np.ndarray]]]:
        """
        Convert tensor dictionary to pyLOM HDF5-compatible format.

        Args:
            attr_dict (Dict[str, torch.Tensor]): Dictionary of tensor attributes.

        Returns:
            Dict[str, Dict]: pyLOM format dictionary.
        """
        return {
            key: {
                'ndim': value.shape[1] if value.ndim > 1 else 1,
                'value': value.cpu().numpy()
            }
            for key, value in attr_dict.items()
        }


    @staticmethod
    def _from_pyLOM_format(
        edge_index: np.ndarray,
        nodeFeatrDict: Dict[str, Dict[str, np.ndarray]],
        edgeFeatrDict: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict:
        """
        Convert pyLOM HDF5 format back to constructor arguments.

        Returns:
            Dict: Keyword arguments for cls.__init__
        """
        return {
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "node_features_dict": {k: torch.tensor(v['value']) for k, v in nodeFeatrDict.items()},
            "edge_features_dict": {k: torch.tensor(v['value']) for k, v in edgeFeatrDict.items()},
        }

    @cr('Graph._compute_node_features_dict')
    @staticmethod
    def _compute_node_features_dict(mesh: Mesh) -> Dict[str, torch.Tensor]:
        r'''Computes the node attributes of Graph as described in
            Hines, D., & Bekemeyer, P. (2023). Graph neural networks for the prediction of aircraft surface pressure distributions.
            Aerospace Science and Technology, 137, 108268.
            https://doi.org/10.1016/j.ast.2023.108268

        Args:
            mesh (Mesh): A RANS mesh in pyLOM format.
        Returns:
            Dict[str, torch.Tensor]: Node attributes of the graph.
        '''
        # Get the cell centers
        xyzc = mesh.xyzc
        # Get the surface normals
        surface_normals = mesh.normal
        
        node_features_dict = {'xyz': torch.tensor(xyzc, dtype=torch.float32), 'normals': torch.tensor(surface_normals, dtype=torch.float32)}

        return node_features_dict

    @cr('Graph._compute_edge_index_and_attr_dict')
    @staticmethod
    def _compute_edge_index_and_attr_dict(mesh: Mesh) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r'''Computes the edge index and attributes of Graph as described in
            Hines, D., & Bekemeyer, P. (2023). Graph neural networks for the prediction of aircraft surface pressure distributions.
            Aerospace Science and Technology, 137, 108268.
            https://doi.org/10.1016/j.ast.2023.108268
        Args:
            mesh (Mesh): A RANS mesh in pyLOM format.
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Edge index and attributes of the graph.
        '''
        # Check whether the cells are 2D
        if not np.all(np.isin(mesh.eltype, [2, 3, 4, 5])):
            raise ValueError("The mesh must contain only 2D cells in order to compute the wall normals.")
        

        # Dictionary that maps each edge to the cells that share it
        edge_dict = edge_to_cells(mesh.connectivity)
        # List storing directed edges in the dual graph
        edge_list = []
        # List to store the wall normals.
        wall_normals_list = []

        # Iterate over each cell
        for i, cell_id in enumerate(range(mesh.ncells)):
            cell_normal = mesh.normal[cell_id]
            cell_nodes = mesh.connectivity[cell_id]
            nodes_xyz = mesh.xyz[cell_nodes]  # Get the nodes of the cell

            cell_edges, cell_wall_normals = wall_normals(cell_nodes, nodes_xyz, cell_normal)  # Compute the edge normals of the cell
            
            # Directed dual edges: tuples of the form (cell_id, neighbor_id)
            dual_edges = [
                (cell_id, (edge_dict[edge] - {cell_id}).pop()) if len(edge_dict[edge]) == 2 else None # If the edge is not a boundary edge, get the neighbor cell
                for edge in cell_edges
            ]

            edge_list.extend(dual_edges)
            wall_normals_list.extend(cell_wall_normals)

            if i%1e5 == 0:
                print(f"Processing mesh. {i} cells out of {mesh.ncells} processed.")

        # Remove the wall normals and dual edges at the boundary walls
        edge_list, wall_normals_list = zip(*[
                (x, y) for x, y in zip(edge_list, wall_normals_list) if x is not None
            ])

        edge_index_np = np.array(edge_list, dtype=np.int64).T  # Convert to numpy array and transpose
        wall_normals_tensor = torch.tensor(wall_normals_list, dtype=torch.float32)  # Convert to torch tensor

        # Compute the rest of the edge_attributes
        # Get the cell centers
        xyzc = mesh.xyzc
        # Get the edge coordinates
        c_i = xyzc[edge_index_np[0, :]]
        c_j = xyzc[edge_index_np[1, :]]
        d_ij = c_j - c_i
        # Transform to spherical coordinates
        r = np.linalg.norm(d_ij, axis=1)           # Distance ||x_i - x_j||  
        theta = np.arccos(d_ij[:, 2] / r)          # Angle from z-axis
        phi = np.arctan2(d_ij[:, 1], d_ij[:, 0])   # Azimuthal angle in xy-plane

        r = torch.from_numpy(r).float()
        theta = torch.from_numpy(theta).float()
        phi = torch.from_numpy(phi).float()
        
        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)
        edge_features_dict = {'edges_spherical': torch.stack((r, theta, phi), dim=1),
                           'wall_normals': wall_normals_tensor}

        return edge_index, edge_features_dict

    def node_attr(self):
        """
        [DEPRECATED] Use `node_features_dict` instead.

        This method is retained for backward compatibility. It returns the dictionary of node attributes,
        where each entry corresponds to a named tensor of shape [num_nodes, feature_dim].

        Note:
            This method will be removed in a future version. Please use `graph.node_features_dict` instead.

        Example:
            >>> g.node_attr()  # Deprecated
            >>> g.node_features_dict['xyz']  # Preferred
        """
        warnings.warn(
            "`node_attr()` is deprecated. Use `graph.node_features_dict` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.node_features_dict



    def edge_attr(self):
        """
        [DEPRECATED] Use `edge_features_dict` instead.

        This method is retained for backward compatibility. It returns the dictionary of edge attributes,
        where each entry corresponds to a named tensor of shape [num_edges, feature_dim].

        Note:
            This method will be removed in a future version. Please use `graph.edge_features_dict` instead.

        Example:
            >>> g.edge_attr()  # Deprecated
            >>> g.edge_features_dict['wall_normals']  # Preferred
        """
        warnings.warn(
            "`edge_attr()` is deprecated. Use `graph.edge_features_dict` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.edge_features_dict


    @cr('Graph.filter')
    def filter(
        self,
        node_mask: Optional[Union[list, torch.Tensor, np.ndarray]] = None,
        node_indices: Optional[Union[list, torch.Tensor, np.ndarray]] = None
    ) -> "Graph":
        """
        ⚠️ Experimental: This method is still under development and has not been thoroughly tested.
        Use with caution and validate outputs manually.

        Return a filtered copy of the graph, keeping only the specified nodes
        and their corresponding edges.

        You can specify either:
            - `node_mask`: a boolean mask of shape [num_nodes]
            - `node_indices`: a list or tensor of node indices to keep

        Args:
            node_mask (Union[list, torch.Tensor, np.ndarray], optional):
                Boolean mask indicating which nodes to keep.
            node_indices (Union[list, torch.Tensor, np.ndarray], optional):
                List of node indices to keep.

        Returns:
            Graph: A new Graph instance containing only the selected nodes and edges.

        Raises:
            ValueError: If neither or both of `node_mask` and `node_indices` are provided.
        """
        if node_mask is None and node_indices is None:
            raise ValueError("Either node_mask or node_indices must be provided.")
        if node_mask is not None and node_indices is not None:
            raise ValueError("Only one of node_mask or node_indices must be provided.")

        if node_indices is not None:
            node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            node_mask[node_indices] = True
        else:
            node_mask = torch.as_tensor(node_mask, dtype=torch.bool)

        # Mapping from old node index to new index
        idx_map = torch.full((self.num_nodes,), -1, dtype=torch.long)
        idx_map[node_mask] = torch.arange(node_mask.sum())

        # Filter node features
        new_node_features_dict = {
            k: v[node_mask] for k, v in self.node_features_dict.items()
        }

        # Identify edges where both source and target nodes are kept
        src, dst = self.edge_index
        edge_mask = node_mask[src] & node_mask[dst]
        new_edge_index = idx_map[self.edge_index[:, edge_mask]]

        # Filter edge features
        new_edge_features_dict = {
            k: v[edge_mask] for k, v in self.edge_features_dict.items()
        }

        return Graph(
            edge_index=new_edge_index,
            node_features_dict=new_node_features_dict,
            edge_features_dict=new_edge_features_dict,
            device=self.edge_index.device
        )
