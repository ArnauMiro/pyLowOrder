#!/usr/bin/env python
#
# Minimal graph example:
#   1) Load DATA/CYLINDER.h5
#   2) Read mesh (group MESH) as pyLOM.Mesh
#   3) Build Graph from mesh
#   4) Print basic graph stats
#   5) Save graph as CYLINDER_GRAPH.h5 in the same folder
#
# Last revision: 23/04/2026

import pyLOM, pyLOM.NN

device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Parameters
DATAFILE  = './DATA/CYLINDER.h5'
GRAPHFILE = './DATA/CYLINDER_GRAPH.h5'

# Load mesh
mesh = pyLOM.Mesh.load(DATAFILE)

# Generate graph from mesh
graph = pyLOM.NN.Graph.from_pyLOM_mesh(mesh=mesh,device=device)
graph.validate()

pyLOM.pprint(0,f"Mesh file: {DATAFILE}")
pyLOM.pprint(0,f"Graph nodes: {int(graph.num_nodes)}")
pyLOM.pprint(0,f"Graph edges: {int(graph.num_edges)}")
pyLOM.pprint(0,f"Node features: {list(graph.node_features_dict.keys())}")
pyLOM.pprint(0,f"Edge features: {list(graph.edge_features_dict.keys())}")

mesh.save(GRAPHFILE,mode="w")
graph.save(GRAPHFILE,mode="a")
print(f"Saved graph: {GRAPHFILE}")
