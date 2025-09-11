#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines.
#
# Last rev: 02/10/2024

# Built-in modules
import os
from typing import Any, Optional, Tuple, Union, Dict
import warnings
import hashlib

# Third-party libraries
import numpy as np
import torch
from torch_geometric.data import Data

# Local modules
from .. import DEVICE
from ... import io, cr
from ...mesh import Mesh
from ...vmmath.geometric import edge_to_cells, wall_normals
from ...utils import raiseError






class Graph(Data):
    def __init__(
        self,
        edge_index: torch.Tensor,
        x_dict: Dict[str, torch.Tensor],
        edge_attr_dict: Dict[str, torch.Tensor],
        device: Union[str, torch.device] = None,
        # **custom_attr_dict: Any # Not yet implemented.
    ):
        r'''
        Initialize the Graph object. Node and edge attributes are stacked separately along dimension 1 for use in GNNs.

        Args:
            edge_index (torch.Tensor): Edge connectivity in COO format [2, num_edges].
            x_dict (Dict[str, torch.Tensor]): Dict of node features with shape [num_nodes, feature_dim] per entry.
            edge_attr_dict (Dict[str, torch.Tensor]): Dict of edge features with shape [num_edges, feature_dim] per entry.
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
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}
        # custom_attr_dict = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in custom_attr_dict.items()} # Not yet implemented.

        # Concatenate node/edge attributes
        x = torch.cat(list(x_dict.values()), dim=1) if x_dict else None
        edge_attr = torch.cat(list(edge_attr_dict.values()), dim=1) if edge_attr_dict else None

        # Build kwargs for Data
        data_kwargs = {
            'edge_index': edge_index,
            'x': x,
            'edge_attr': edge_attr,
            'num_nodes': next(iter(x_dict.values())).shape[0] if x_dict else None
            # **custom_attr_dict,
        }

        super().__init__(**data_kwargs)

        # Set device
        self.device = device
        
        # Register individual attributes for user-friendly access
        for k, v in x_dict.items():
            setattr(self, k, v)
        for k, v in edge_attr_dict.items():
            setattr(self, k, v)

        # store raw attribute dicts (for i/o operations)
        self.x_dict = x_dict
        self.edge_attr_dict = edge_attr_dict
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
        self._validate_x()
        self._validate_edge_attr()
        self._validate_edge_index()
        return self

    def _validate_x(self):
        assert isinstance(self.x_dict, dict), "x_dict must be a dictionary"
        assert len(set(self.x_dict)) == len(self.x_dict), "Duplicate keys in x_dict"
        
        # Check shapes
        for k, v in self.x_dict.items():
            assert isinstance(v, torch.Tensor), f"Node attribute '{k}' must be a tensor"
            assert v.shape[0] == self.num_nodes, f"Node attribute '{k}' has wrong number of nodes: expected {self.num_nodes}, got {v.shape[0]}"
            assert not torch.isnan(v).any(), f"Node attribute '{k}' contains NaNs"

        # Validate concatenation
        x_cat = torch.cat(list(self.x_dict.values()), dim=-1)
        assert self.x.shape == x_cat.shape, "x shape mismatch with concatenated x_dict"
        assert torch.allclose(self.x, x_cat, atol=1e-6), "x do not match concatenated x_dict"

    def _validate_edge_attr(self):
        assert isinstance(self.edge_attr_dict, dict), "edge_attr_dict must be a dictionary"
        assert len(set(self.edge_attr_dict)) == len(self.edge_attr_dict), "Duplicate keys in edge_attr_dict"
        
        for k, v in self.edge_attr_dict.items():
            assert isinstance(v, torch.Tensor), f"Edge attribute '{k}' must be a tensor"
            assert v.shape[0] == self.num_edges, f"Edge attribute '{k}' has wrong number of edges: expected {self.num_edges}, got {v.shape[0]}"
            assert not torch.isnan(v).any(), f"Edge attribute '{k}' contains NaNs"

        edge_attr_cat = torch.cat(list(self.edge_attr_dict.values()), dim=-1)
        assert self.edge_attr.shape == edge_attr_cat.shape, "edge_attr shape mismatch with concatenated edge_attr_dict"
        assert torch.allclose(self.edge_attr, edge_attr_cat, atol=1e-6), "edge_attr do not match concatenated edge_attr_dict"

    def _validate_edge_index(self):
        assert isinstance(self.edge_index, torch.Tensor), "edge_index must be a tensor"
        assert self.edge_index.dtype in (torch.int32, torch.int64), "edge_index must be integer typed"
        assert self.edge_index.shape[0] == 2, "edge_index must have shape [2, num_edges]"
        assert self.edge_index.shape[1] == self.num_edges, "edge_index second dimension must match number of edges"
        assert int(self.edge_index.max()) < int(self.num_nodes), "edge_index contains invalid node indices"
        # torch.isnan() does not support integer tensors; use a bounds check instead
        assert (self.edge_index >= 0).all(), "edge_index contains negative indices"


    @classmethod
    @cr('Graph.from_pyLOM_mesh')
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
        x_dict = cls._compute_x_dict(mesh)  # Get the node attributes
        edge_index, edge_attr_dict = cls._compute_edge_index_and_attr_dict(mesh)  # Get the edge attributes

        graph = cls(
            edge_index=edge_index,
            x_dict = x_dict,
            edge_attr_dict=edge_attr_dict,
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
            node_dict = self._to_pyLOM_format(self.x_dict)
            edge_dict = self._to_pyLOM_format(self.edge_attr_dict)
            io.h5_save_graph_serial(
                fname,
                num_nodes=self.num_nodes,
                num_edges=self.num_edges,
                edge_index=self.edge_index.cpu().numpy(),
                xDict=node_dict,
                edgeAttrDict=edge_dict,
                **kwargs
            )
        elif fmt in ['pt', 'pkl']:
            torch.save(self, fname)
        else:
            raiseError(f"Unsupported file format: {fmt}")

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
            num_nodes, num_edges, edge_index, xDict, edgeAttrDict = io.h5_load_graph_serial(fname)
            init_kwargs = cls._from_pyLOM_format(edge_index, xDict, edgeAttrDict)

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
            raiseError(f"Unsupported file format: {fmt}")


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
        xDict: Dict[str, Dict[str, np.ndarray]],
        edgeAttrDict: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict:
        """
        Convert pyLOM HDF5 format back to constructor arguments.

        Returns:
            Dict: Keyword arguments for cls.__init__
        """
        return {
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "x_dict": {k: torch.tensor(v['value']) for k, v in xDict.items()},
            "edge_attr_dict": {k: torch.tensor(v['value']) for k, v in edgeAttrDict.items()},
        }

    @staticmethod
    def _canonical_edge(edge: Union[np.ndarray, list, tuple]) -> tuple[int, ...]:
        """
        Return an immutable, undirected, canonical representation of an edge (or face).
        Converts any array-like into a *deduplicated* sorted tuple of ints, safe as a dict key.

        Notes
        -----
        - Dedup is critical because some providers (e.g., wall_normals) may repeat node ids.
        - For 2D surfaces, edges should reduce to exactly 2 unique nodes.
        """
        e = np.asarray(edge).astype(int).ravel()
        u = np.unique(e)                # <-- remove duplicates
        return tuple(sorted(u.tolist()))

    @staticmethod
    def _edge_key_from_local_or_global(
        edge: Union[np.ndarray, list, tuple],
        cell_nodes: Optional[np.ndarray] = None,
    ) -> Optional[tuple[int, int]]:
        """
        Build a canonical, hashable, undirected key for an edge using *global* node ids.

        If `cell_nodes` is provided and `edge` looks like local indices (i.e., max(edge) < len(cell_nodes)),
        the function converts local -> global via `cell_nodes[edge]`. Then it deduplicates and sorts.

        Returns:
            (u, v) as an increasing tuple of two global node ids, or None if the edge is degenerate.
        """
        e = np.asarray(edge).ravel().astype(int)

        # Heuristic: treat as local indices if they fit within the local array bounds
        if cell_nodes is not None and e.size > 0 and e.max(initial=-1) < len(cell_nodes):
            # map local -> global
            g = np.asarray(cell_nodes, dtype=int)[e]
        else:
            # assume already global
            g = e

        # deduplicate and sort to remove orientation ambiguity and repeated ids
        uniq = np.unique(g)
        if uniq.size != 2:
            return None  # degenerate (boundary collapse or malformed edge)
        u, v = sorted(uniq.tolist())
        return (u, v)


    @cr('Graph._compute_x_dict')
    @staticmethod
    def _compute_x_dict(mesh: Mesh) -> Dict[str, torch.Tensor]:
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
        
        x_dict = {'xyz': torch.tensor(xyzc, dtype=torch.float32), 'normals': torch.tensor(surface_normals, dtype=torch.float32)}

        return x_dict

    @cr('Graph._compute_edge_index_and_attr_dict')
    @staticmethod
    def _compute_edge_index_and_attr_dict(mesh: Mesh) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Compute dual-graph edge index and attributes from a pyLOM surface mesh.

        Design:
        - Decouple topology from geometry:
        (1) Build global edge→cells incidence from connectivity.
        (2) For each cell, compute wall normals (Cython) and record them per (cell_id, global_edge).
        (3) For each interior global edge with {a,b}, emit directed dual edges a→b and b→a,
            attaching the wall-normal as seen from the source cell.

        Robustness:
        - Enforce Cython dtypes/contiguity (intc/float64).
        - Resolve local vs. global edge ambiguity by trying both interpretations.
        """
        import numpy as np
        import torch
        from collections import defaultdict

        # --- guards
        if not np.all(np.isin(mesh.eltype, [2, 3, 4, 5])):
            raiseError("The mesh must contain only 2D cells to compute wall normals.")

        def canon(u: int, v: int) -> tuple[int, int]:
            return (u, v) if u < v else (v, u)

        def resolve_global_edge(edge, cell_nodes, incidence) -> Optional[tuple[int, int]]:
            """Return a canonical GLOBAL (u,v) for a raw edge that may be global or local.
            Try global first; if not found, try local→global mapping.
            """
            e = np.asarray(edge, dtype=np.int64).ravel()
            if e.size < 2:
                return None

            # A) assume global
            a = canon(int(e[0]), int(e[1]))
            if a in incidence:
                return a

            # B) assume local positions
            cn = np.asarray(cell_nodes, dtype=np.int64).ravel()
            if int(e.max(initial=-1)) < len(cn):
                b = canon(int(cn[int(e[0])]), int(cn[int(e[1])]))
                if b in incidence:
                    return b

            return None

        # --- 1) topology: build incidence from connectivity (GLOBAL ids)
        incidence: dict[tuple[int, int], set[int]] = defaultdict(set)
        for cell_id, cnodes in enumerate(mesh.connectivity):
            cn = np.asarray(cnodes, dtype=np.int64).ravel()
            # polygon ring (assumes cyclic order)
            for a, b in zip(cn, np.roll(cn, -1)):
                incidence[canon(int(a), int(b))].add(cell_id)

        n_interior = sum(1 for s in incidence.values() if len(s) == 2)
        if n_interior == 0:
            raiseError("No interior edges found. Mesh likely not welded (cells do not share node IDs).")

        # --- 2) geometry: per-cell wall normals (Cython expects intc/float64)
        wall_normal_map: dict[tuple[int, tuple[int, int]], np.ndarray] = {}
        for cell_id in range(mesh.ncells):
            # dtypes/contiguity para Cython
            cell_nodes_intc = np.ascontiguousarray(mesh.connectivity[cell_id], dtype=np.intc)     # GLOBAL ids pero intc
            nodes_xyz_f64   = np.ascontiguousarray(mesh.xyz[cell_nodes_intc], dtype=np.float64)
            cell_norm_f64   = np.asarray(mesh.normal[cell_id], dtype=np.float64)

            # Cálculo geométrico
            # Devuelve listas alineadas por índice: i ↔ arista (i, i+1)
            _, cell_wall_normals = wall_normals(cell_nodes_intc, nodes_xyz_f64, cell_norm_f64)

            # Usamos SIEMPRE el par global tomado de connectivity por posición (i, i+1)
            cell_nodes_global = np.asarray(mesh.connectivity[cell_id], dtype=np.int64)
            n = int(cell_nodes_global.size)
            for i, wn in enumerate(cell_wall_normals):
                u = int(cell_nodes_global[i])
                v = int(cell_nodes_global[(i + 1) % n])
                gk = (u, v) if u < v else (v, u)  # clave canónica global
                # Solo guardamos si la arista existe en la incidencia (interior o borde)
                if gk in incidence:
                    wall_normal_map[(cell_id, gk)] = np.asarray(wn, dtype=float)


        # --- 3) build directed dual edges + attach wall normals (source-view)
        pairs: list[tuple[int, int]] = []
        wn_list: list[np.ndarray] = []
        for gk, cells in incidence.items():
            if len(cells) != 2:
                continue  # boundary
            a, b = tuple(cells)
            # a -> b
            pairs.append((a, b))
            wn_list.append(wall_normal_map.get((a, gk), np.zeros(3, dtype=float)))
            # b -> a
            pairs.append((b, a))
            wn_list.append(wall_normal_map.get((b, gk), np.zeros(3, dtype=float)))

        if not pairs:
            raiseError("No dual edges were constructed. Check wall_normals and edge mapping.")

        edge_index_np = np.asarray(pairs, dtype=np.int64).T  # [2, E]
        wall_normals_tensor = torch.tensor(np.asarray(wn_list), dtype=torch.float32)

        # --- edge geometric attributes (spherical direction i->j)
        xyzc = mesh.xyzc
        c_i = xyzc[edge_index_np[0, :]]
        c_j = xyzc[edge_index_np[1, :]]
        d_ij = c_j - c_i

        r = np.linalg.norm(d_ij, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            theta = np.arccos(np.clip(d_ij[:, 2] / np.where(r == 0.0, 1.0, r), -1.0, 1.0))
        phi = np.arctan2(d_ij[:, 1], d_ij[:, 0])

        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)
        edge_attr_dict = {
            'edges_spherical': torch.stack((torch.from_numpy(r).float(),
                                            torch.from_numpy(theta).float(),
                                            torch.from_numpy(phi).float()), dim=1),
            'wall_normals': wall_normals_tensor
        }
        return edge_index, edge_attr_dict





    def node_attr(self):
        """
        [DEPRECATED] Use `x_dict` instead.

        This method is retained for backward compatibility. It returns the dictionary of node attributes,
        where each entry corresponds to a named tensor of shape [num_nodes, feature_dim].

        Note:
            This method will be removed in a future version. Please use `graph.x_dict` instead.

        Example:
            >>> g.node_attr()  # Deprecated
            >>> g.x_dict['xyz']  # Preferred
        """
        warnings.warn(
            "`node_attr()` is deprecated. Use `graph.x_dict` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.x_dict


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
            raiseError("Either node_mask or node_indices must be provided.")
        if node_mask is not None and node_indices is not None:
            raiseError("Only one of node_mask or node_indices must be provided.")

        if node_indices is not None:
            node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            node_mask[node_indices] = True
        else:
            node_mask = torch.as_tensor(node_mask, dtype=torch.bool)

        # Mapping from old node index to new index
        idx_map = torch.full((self.num_nodes,), -1, dtype=torch.long)
        idx_map[node_mask] = torch.arange(node_mask.sum())

        # Filter node features
        new_x_dict = {
            k: v[node_mask] for k, v in self.x_dict.items()
        }

        # Identify edges where both source and target nodes are kept
        src, dst = self.edge_index
        edge_mask = node_mask[src] & node_mask[dst]
        new_edge_index = idx_map[self.edge_index[:, edge_mask]]

        # Filter edge features
        new_edge_attr_dict = {
            k: v[edge_mask] for k, v in self.edge_attr_dict.items()
        }

        return Graph(
            edge_index=new_edge_index,
            x_dict=new_x_dict,
            edge_attr_dict=new_edge_attr_dict,
            device=self.edge_index.device
        )

    def fingerprint(self) -> Dict[str, Any]:
        """
        Return a deterministic fingerprint of the graph topology and key shapes.

        Notes
        -----
        - Hash includes only connectivity (edge_index), not floating values,
          so it is stable across dtype/device conversions.
        """
        if not hasattr(self, "num_nodes") or not hasattr(self, "edge_index"):
            raiseError("Graph must have 'num_nodes' and 'edge_index' to be fingerprinted.")

        num_nodes = int(self.num_nodes)
        ei = self.edge_index
        if not isinstance(ei, torch.Tensor) or ei.ndim != 2 or ei.size(0) != 2:
            raiseError("Graph 'edge_index' must be a 2 x E tensor.")

        num_edges = int(ei.size(1))

        # Hash connectivity on CPU, int64, contiguous
        ei_cpu = ei.to(dtype=torch.int64, device="cpu", non_blocking=False).contiguous()
        h = hashlib.sha256(ei_cpu.numpy().tobytes()).hexdigest()

        x_shape: Optional[tuple] = tuple(self.x.shape) if getattr(self, "x", None) is not None else None
        eattr_shape: Optional[tuple] = tuple(self.edge_attr.shape) if getattr(self, "edge_attr", None) is not None else None

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "edge_index_sha256": h,
            "x_shape": x_shape,
            "edge_attr_shape": eattr_shape,
        }