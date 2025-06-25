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

# Third-party libraries
import numpy as np
import torch
from torch_geometric.data import Data

# Local modules
from .. import DEVICE
from ... import io
from ...mesh import Mesh
from ...utils.cr import cr
from ...vmmath.geometric import edge_to_cells, wall_normals






class Graph(Data):
    def __init__(
        self,
        edge_index: torch.Tensor,
        node_attrs: Dict[str, torch.Tensor],
        edge_attrs: Dict[str, torch.Tensor],
        device: Union[str, torch.device] = None,
        # **custom_attrs: Any # Not yet implemented.
    ):
        r'''
        Initialize the Graph object. Node and edge attributes are stacked separately along dimension 1 for use in GNNs.

        Args:
            edge_index (torch.Tensor): COO format edge index (shape [2, num_edges]).
            node_attrs (Dict[str, torch.Tensor]): Dict of node attributes [num_nodes, attr_dim].
            edge_attrs (Dict[str, torch.Tensor]): Dict of edge attributes [num_edges, attr_dim].
            device (Union[str, torch.device], optional): Torch device. Defaults to CUDA if available.
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
        node_attrs = {k: v.to(device) for k, v in node_attrs.items()}
        edge_attrs = {k: v.to(device) for k, v in edge_attrs.items()}
        # custom_attrs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in custom_attrs.items()} # Not yet implemented.

        # Concatenate node/edge attributes
        x = torch.cat(list(node_attrs.values()), dim=1) if node_attrs else None
        edge_attr = torch.cat(list(edge_attrs.values()), dim=1) if edge_attrs else None

        # Build kwargs for Data
        data_kwargs = {
            'edge_index': edge_index,
            'x': x,
            'edge_attr': edge_attr,
            'num_nodes': next(iter(node_attrs.values())).shape[0] if node_attrs else None
            # **custom_attrs,
        }

        super().__init__(**data_kwargs)

        # store raw attribute dicts (for i/o operations)
        self.node_attrs_dict = node_attrs
        self.edge_attrs_dict = edge_attrs
        # self.custom_attrs = custom_attrs # Not yet implemented.

        self.validate()

    def validate(self) -> None:
        assert self.edge_index.shape[0] == 2
        assert self.edge_index.dtype == torch.long

        for k, v in self.node_attrs_dict.items():
            assert v.shape[0] == self.num_nodes, f"Node attribute '{k}' has inconsistent length: {v.shape[0]} vs expected {self.num_nodes}"
        for k, v in self.edge_attrs_dict.items():
            assert v.shape[0] == self.edge_index.shape[1], f"Edge attribute '{k}' has inconsistent length: {v.shape[0]} vs expected {self.edge_index.shape[1]}"
        
        if self.edge_attr is not None:
            expected_dim = sum(v.shape[1] if v.ndim > 1 else 1 for v in self.edge_attrs_dict.values())
            assert self.edge_attr.shape[1] == expected_dim
        
        assert all(isinstance(k, str) and k.strip() for k in self.node_attrs_dict), "All node attribute keys must be non-empty strings."
        assert all(isinstance(k, str) and k.strip() for k in self.edge_attrs_dict), "All edge attribute keys must be non-empty strings."
        
        for name, tensor in self.node_attrs_dict.items():
            assert torch.isfinite(tensor).all(), f"Node attribute '{name}' contains NaN or inf."
        for name, tensor in self.edge_attrs_dict.items():
            assert torch.isfinite(tensor).all(), f"Edge attribute '{name}' contains NaN or inf."


    @classmethod
    def from_pyLOM_mesh(cls,
                        mesh: Mesh,
                        device: Optional[Union[str, torch.device]] = None,
                        # **custom_attrs: Dict[str, Any]  # Not yet implemented.
                        ) -> "Graph":
        r"""
        Create a Graph object from a pyLOM Mesh object. This method computes the node attributes and edge index/attributes from the mesh.
        Args:
            mesh (Mesh): A mesh object in pyLOM format.
            device (Optional[Union[str, torch.device]]): The device to use for the graph.
        Returns:
            Graph: A Graph object with the computed node attributes and edge index/attributes.
        """
        node_attrs_dict = cls._compute_node_attrs(mesh)  # Get the node attributes
        edge_index, edge_attrs_dict = cls._compute_edge_index_and_attrs(mesh)  # Get the edge attributes

        graph = cls(
            edge_index=edge_index,
            node_attrs = node_attrs_dict,
            edge_attrs=edge_attrs_dict,
            device=device,
            # **custom_attrs # Not yet implemented.
            )

        return graph

    @cr('Graph.save')
    def save(self,fname,**kwargs) -> None:
        '''
        Store the graph in a h5 file, pyLOM style.
        '''
        # Set default parameters
        if not 'mode' in kwargs.keys():        kwargs['mode']        = 'w' if not os.path.exists(fname) else 'a'
        # Append or save
        edge_index = self.edge_index.cpu().numpy()
        node_save_dict = self._to_pyLOM_format(self.node_attrs_dict)
        edge_save_dict = self._to_pyLOM_format(self.edge_attrs_dict)
        if not kwargs.pop('append',False):
            io.h5_save_graph_serial(fname,edge_index,node_save_dict,edge_save_dict,**kwargs)
        else:
            io.h5_append_graph_serial(fname,edge_index,node_save_dict,edge_save_dict,**kwargs)

    @classmethod
    def load(
        cls,
        fname: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Graph":
        r'''
        Load a graph from a h5 file, pyLOM style.
        '''
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Graph file {fname} not found.")
        if not fname.endswith('.h5'):
            raise ValueError(f"Graph file {fname} must be a .h5 file.")

        edge_index,node_attrs_dict,edge_attrs_dict = io.h5_load_graph_serial(fname)

        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        node_attrs = {key: torch.tensor(value['value'], dtype=torch.float32) for key, value in node_attrs_dict.items()}
        edge_attrs = {key: torch.tensor(value['value'], dtype=torch.float32) for key, value in edge_attrs_dict.items()}  

        return cls(edge_index=edge_index,
                   node_attrs=node_attrs,
                   edge_attrs=edge_attrs,
                   device=device,
                   )

    def save_pt(self, fname: str) -> None:
        r'''
        Save the graph to a PyTorch .pt file.
        Args:
            fname (str): The filename to save the graph to.
        '''
        if not fname.endswith('.pt'):
            raise ValueError(f"Graph file {fname} must be a .pt file.")
        outdir = os.path.dirname(fname)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        torch.save(self, fname)

    @classmethod
    def load_pt(cls, fname: str, device: Union[str, torch.device]) -> "Graph":
        r'''
        Load a graph from a PyTorch .pt file.
        Args:
            fname (str): The filename to load the graph from.
        Returns:
            Graph: The loaded graph object.
        '''
        if not fname.endswith('.pt'):
            raise ValueError(f"Graph file {fname} must be a .pt file.")
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Graph file {fname} not found.")
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please use CPU instead.")
        return torch.load(fname, map_location=str(device))


    @staticmethod
    def _to_pyLOM_format(attrs_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Union[int, np.ndarray]]]:
        r'''
        Converts a dictionary of numpy arrays to a dictionary of dictionaries with the shape and value keys.
        This is used to save the node and edge attributes in pyLOM format.
        '''
        save_dict = {}
        for key, value in attrs_dict.items():
            save_dict[key] = {
                'ndim': value.shape[1] if len(value.shape) > 1 else 1,
                'value': value.cpu().numpy()
            }
        return save_dict

    @staticmethod
    def _compute_node_attrs(mesh: Mesh) -> Dict[str, torch.Tensor]:
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
        
        node_attrs_dict = {'xyz': torch.tensor(xyzc, dtype=torch.float32), 'normals': torch.tensor(surface_normals, dtype=torch.float32)}

        return node_attrs_dict


    @staticmethod
    def _compute_edge_index_and_attrs(mesh: Mesh) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        r = np.linalg.norm(d_ij, axis=1)
        theta = np.arccos(d_ij[:, 2] / r)
        phi = np.arctan2(d_ij[:, 1], d_ij[:, 0])

        r = torch.from_numpy(r).float()
        theta = torch.from_numpy(theta).float()
        phi = torch.from_numpy(phi).float()
        
        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)
        edge_attrs_dict = {'edges_spherical': torch.stack((r, theta, phi), dim=1),
                           'wall_normals': wall_normals_tensor}

        return edge_index, edge_attrs_dict

    # def filter(self,
    #     node_mask: Optional[Union[list, torch.Tensor, np.ndarray]]=None,
    #     node_indices: Optional[Union[list, torch.Tensor, np.ndarray]]=None
    # ):
    #     r'''
    #     Filter graph by providing either a boolean mask or a list of node indices to keep.

    #     Args:
    #         node_mask: Boolean mask to filter nodes.
    #         node_indices: List of node indices to keep.
    #     '''
        

    #     if node_mask is None and node_indices is None:
    #         raise ValueError("Either node_mask or node_indices must be provided.")
    #     elif node_mask is not None and node_indices is not None:
    #         raise ValueError("Only one of node_mask or node_indices must be provided.")
    #     elif node_mask is not None:
    #         node_mask = torch.tensor(node_mask, dtype=torch.bool)
    #     elif node_indices is not None:
    #         node_mask = torch.zeros(self.x.shape[0], dtype=torch.bool)
    #         node_mask[node_indices] = True

    #     for attr in self.node_attrs():
    #         if getattr(self, attr) is not None:
    #             setattr(self, attr, getattr(self, attr)[node_mask])

    #     self.edge_attr = self.edge_attr[torch.logical_and(node_mask[self.edge_index[0]], node_mask[self.edge_index[1]])]
    #     self.edge_index = self.edge_index[:, torch.logical_and(node_mask[self.edge_index[0]], node_mask[self.edge_index[1]])]
    #     self.edge_index -= torch.min(self.edge_index)