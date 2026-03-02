#!/usr/bin/env python
"""
Minimal graph example:
1) Load Testsuite/DATA/CYLINDER.h5
2) Read mesh (group MESH) as pyLOM.Mesh
3) Build Graph from mesh
4) Print basic graph stats
5) Save graph as CYLINDER_GRAPH.h5 in the same folder
"""

from pathlib import Path
import h5py

from pyLOM import Mesh
from pyLOM.NN.gns.graph import Graph


repo_root = Path(__file__).resolve().parents[2]
mesh_path = (repo_root / "Testsuite" / "DATA" / "CYLINDER.h5").resolve()
graph_path = mesh_path.with_name("CYLINDER_GRAPH.h5")

with h5py.File(mesh_path, "r") as f:
    if "MESH" not in f:
        raise RuntimeError(f"Missing 'MESH' group in: {mesh_path}")

mesh = Mesh.load(str(mesh_path))
graph = Graph.from_pyLOM_mesh(mesh=mesh, device="cpu")
graph.validate()

print(f"Mesh file: {mesh_path}")
print(f"Graph nodes: {int(graph.num_nodes)}")
print(f"Graph edges: {int(graph.num_edges)}")
print(f"Node features: {list(graph.node_features_dict.keys())}")
print(f"Edge features: {list(graph.edge_features_dict.keys())}")

graph.save(str(graph_path), mode="w")
print(f"Saved graph: {graph_path}")
