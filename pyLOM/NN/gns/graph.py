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
from ...vmmath.geometric import wall_normals
from ...utils import raiseError



def _to_tensor_dict_strict(features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, arr in features.items():
        a = np.asarray(arr)
        if a.dtype == np.dtype('O') or a.dtype.kind in ('U', 'S'):
            raiseError(f"Non-numeric feature '{key}' with dtype={a.dtype} found in HDF5.")
        a = a.astype(np.float32, copy=False)
        out[key] = torch.as_tensor(a)
    return out


def _canon_pair(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


class Graph(Data):
    def __init__(
        self,
        edge_index: torch.Tensor,
        node_features_dict: Optional[Dict[str, torch.Tensor]] = None,
        edge_features_dict: Optional[Dict[str, torch.Tensor]] = None,
        *,
        device: Union[str, torch.device] = None,
    ):
        r'''
        Initialize the Graph object. Node and edge attributes are stacked separately along dimension 1 for use in GNNs.

        Args:
            edge_index (torch.Tensor): Edge connectivity in COO format [2, num_edges].
            node_features_dict (Dict[str, torch.Tensor], optional): Preferred name for the node-feature dictionary. Shape per entry [num_nodes, feature_dim].
            edge_features_dict (Dict[str, torch.Tensor], optional): Preferred name for the edge-feature dictionary. Shape per entry [num_edges, feature_dim].
            device (Union[str, torch.device], optional): Computation device. Defaults to global DEVICE.
        '''
        if node_features_dict is None:
            raiseError("Missing node feature dictionary. Expected 'node_features_dict'.")
        if edge_features_dict is None:
            raiseError("Missing edge feature dictionary. Expected 'edge_features_dict'.")

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

        # Concatenate node/edge attributes
        x = torch.cat(list(node_features_dict.values()), dim=1) if node_features_dict else None
        edge_attr = torch.cat(list(edge_features_dict.values()), dim=1) if edge_features_dict else None

        # Build kwargs for Data
        data_kwargs = {
            'edge_index': edge_index,
            'x': x,
            'edge_attr': edge_attr,
            'num_nodes': next(iter(node_features_dict.values())).shape[0] if node_features_dict else None
        }

        super().__init__(**data_kwargs)

        # Set device
        self.device = device
        
        # Register individual attributes for user-friendly access
        for k, v in node_features_dict.items():
            setattr(self, k, v)
        for k, v in edge_features_dict.items():
            setattr(self, k, v)

        # store raw attribute dicts (for i/o operations)
        self.node_features_dict = node_features_dict
        self.edge_features_dict = edge_features_dict

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
        x_cat = torch.cat(list(self.node_features_dict.values()), dim=-1)
        assert self.x.shape == x_cat.shape, "x shape mismatch with concatenated node_features_dict"
        assert torch.allclose(self.x, x_cat, atol=1e-6), "x does not match concatenated node_features_dict"

    def _validate_edge_features(self):
        assert isinstance(self.edge_features_dict, dict), "edge_features_dict must be a dictionary"
        assert len(set(self.edge_features_dict)) == len(self.edge_features_dict), "Duplicate keys in edge_features_dict"
        
        for k, v in self.edge_features_dict.items():
            assert isinstance(v, torch.Tensor), f"Edge attribute '{k}' must be a tensor"
            assert v.shape[0] == self.num_edges, f"Edge attribute '{k}' has wrong number of edges: expected {self.num_edges}, got {v.shape[0]}"
            assert not torch.isnan(v).any(), f"Edge attribute '{k}' contains NaNs"

        edge_attr_cat = torch.cat(list(self.edge_features_dict.values()), dim=-1)
        assert self.edge_attr.shape == edge_attr_cat.shape, "edge_attr shape mismatch with concatenated edge_features_dict"
        assert torch.allclose(self.edge_attr, edge_attr_cat, atol=1e-6), "edge_attr does not match concatenated edge_features_dict"

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
        edge_index, edge_features_dict = cls._compute_edge_index_and_features_dict(mesh)  # Get the edge attributes

        graph = cls(
            edge_index=edge_index,
            node_features_dict=node_features_dict,
            edge_features_dict=edge_features_dict,
            device=device,
            )

        return graph

    def save(self, fname: str, mode: Optional[str] = None, **kwargs):
        """
        Persist the graph to disk. Supports .h5, .pt, .pkl.

        HDF5 schema (strict, graph_flat_v2)
        -----------------------------------
        - /GRAPH attrs['schema'] = "graph_flat_v2"
        - /GRAPH/numNodes : i4[1]
        - /GRAPH/numEdges : i4[1]
        - /GRAPH/edgeIndex : i4[2,E]
        - /GRAPH/NODEFEATRS : group
            * attrs['feature_names'] : S[]
            * <feat_name> : float32[N, k_i]
        - /GRAPH/EDGEFEATRS : group
            * attrs['feature_names'] : S[]
            * <feat_name> : float32[E, k_i]

        Parameters
        ----------
        fname : str
            Output file path.
        mode : {'w','a'} or None
            HDF5 file mode when saving to .h5. Use 'a' to preserve other
            root groups and only replace /GRAPH. Ignored for .pt/.pkl. When
            omitted, defaults to 'w' if ``fname`` does not exist and 'a'
            otherwise (matching ``Dataset.save``).

        Notes
        -----
        * Only numeric features are stored under NODEFEATRS/EDGEFEATRS.
        Text metadata must be stored elsewhere (e.g., /GRAPH/METADATA).
        * Only the /GRAPH group is created/overwritten; other root groups
        (e.g., /DATASET) remain untouched.
        """
        fmt = os.path.splitext(fname)[1][1:].lower()

        if fmt == 'h5':
            # Flat numeric dicts
            x_np = {k: v.detach().cpu().numpy().astype(np.float32, copy=False)
                    for k, v in self.node_features_dict.items()}
            e_np = {k: v.detach().cpu().numpy().astype(np.float32, copy=False)
                    for k, v in self.edge_features_dict.items()}

            # Early type check (reject objects/strings)
            for name, arr in list(x_np.items()) + list(e_np.items()):
                dt = np.asarray(arr).dtype
                if dt == np.dtype('O') or dt.kind in ('U', 'S'):
                    raiseError(f"HDF5 requires numeric arrays; feature '{name}' has dtype={dt}.")

            if mode is None:
                mode_kw = 'w' if not os.path.exists(fname) else 'a'
            else:
                mode_kw = mode

            io.h5_save_graph_serial(
                fname=fname,
                num_nodes=int(self.num_nodes),
                num_edges=int(self.num_edges),
                edge_index=self.edge_index.detach().cpu().numpy().astype(_np.int32, copy=False),
                node_features_dict=x_np,
                edge_features_dict=e_np,
                mode=mode_kw,
            )

        elif fmt in ['pt', 'pkl']:
            torch.save(self, fname)
        else:
            raiseError(f"Unsupported file format: {fmt}")


    @classmethod
    def load(cls, fname: str, **kwargs) -> "Graph":
        """
        Load a graph from disk (strict HDF5 loader: graph_flat_v2 only).

        Only the /GRAPH group is inspected; other root groups are ignored.

        Args:
            fname: Path to the HDF5/PKL/PT file.
            device (optional kwarg): target device.

        Returns:
            Graph
        """
        fmt = os.path.splitext(fname)[1][1:].lower()

        if fmt == 'h5':
            num_nodes, num_edges, edge_index, node_features_np, edge_features_np = io.h5_load_graph_serial(fname)

            node_features_dict = _to_tensor_dict_strict(node_features_np)
            edge_features_dict = _to_tensor_dict_strict(edge_features_np)

            device = kwargs.get('device', DEVICE)
            graph = cls(
                edge_index=torch.as_tensor(edge_index, dtype=torch.long, device='cpu'),
                node_features_dict=node_features_dict,
                edge_features_dict=edge_features_dict,
                device=device,
            )

            # Structural checks
            assert int(graph.num_nodes) == int(num_nodes), f"num_nodes mismatch ({graph.num_nodes} vs {num_nodes})"
            assert int(graph.num_edges) == int(num_edges), f"num_edges mismatch ({graph.num_edges} vs {num_edges})"

            return graph

        elif fmt in ['pt', 'pkl']:
            g = torch.load(fname)
            if not isinstance(g, cls):
                raiseError("Loaded object is not a Graph instance.")
            g.validate()
            return g

        else:
            raiseError(f"Unsupported file format: {fmt}")

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
        
        node_features_dict = {
            'xyz': torch.tensor(xyzc, dtype=torch.float32),
            'normals': torch.tensor(surface_normals, dtype=torch.float32)
        }

        return node_features_dict

    @cr('Graph._compute_edge_index_and_features_dict')
    @staticmethod
    def _compute_edge_index_and_features_dict(mesh: Mesh) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Compute dual-graph edge index and attributes from a pyLOM surface mesh.

        Design:
        - Decouple topology from geometry:
        (1) Build global edge→cells incidence from connectivity.
        (2) For each cell, compute wall normals (Cython) and record them per (cell_id, global_edge).
        (3) For each interior global edge with {a,b}, emit directed dual edges a→b and b→a,
            attaching the wall-normal as seen from the source cell.

        Robustness:
        - Enforce Cython dtypes/contiguity (intc/float64).
        """
        # --- guards
        if not np.all(np.isin(mesh.eltype, [2, 3, 4, 5])):
            raiseError("The mesh must contain only 2D cells to compute wall normals.")
        # --- 1) topology: build incidence from connectivity (GLOBAL ids)
        incidence: dict[tuple[int, int], set[int]] = defaultdict(set)
        for cell_id, cnodes in enumerate(mesh.connectivity):
            cn = np.asarray(cnodes, dtype=np.int64).ravel()
            # polygon ring (assumes cyclic order)
            for a, b in zip(cn, np.roll(cn, -1)):
                incidence[_canon_pair(int(a), int(b))].add(cell_id)

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
                gk = _canon_pair(u, v)  # clave canónica global
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
        if d_ij.shape[1] == 2:
            # 2D meshes: keep spherical contract [r, theta, phi] with theta=pi/2.
            theta = np.full(r.shape, np.pi / 2.0, dtype=r.dtype)
            phi = np.arctan2(d_ij[:, 1], d_ij[:, 0])
        elif d_ij.shape[1] == 3:
            with np.errstate(invalid='ignore', divide='ignore'):
                theta = np.arccos(np.clip(d_ij[:, 2] / np.where(r == 0.0, 1.0, r), -1.0, 1.0))
            phi = np.arctan2(d_ij[:, 1], d_ij[:, 0])
        else:
            raiseError(
                f"Unsupported mesh center dimensionality for edges_spherical: {d_ij.shape[1]}. "
                "Expected 2 or 3."
            )

        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)
        edge_features_dict = {
            'edges_spherical': torch.stack((torch.from_numpy(r).float(),
                                            torch.from_numpy(theta).float(),
                                            torch.from_numpy(phi).float()), dim=1),
            'wall_normals': wall_normals_tensor
        }
        return edge_index, edge_features_dict





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
