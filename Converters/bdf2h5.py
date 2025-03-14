import pyLOM
import numpy as np
from pyNastran.bdf.bdf import BDF

# these paths need to be set
BDFFILE = ""
OUTFILE = ""

BDF2ELTYPE = {
    "CTRIA3": 2,  # Triangular cell
    "CQUAD4": 3,  # Quadrangular cell
}

# Load and read BDF file
model = BDF()
model.read_bdf(BDFFILE, xref=False)

# Create needed variables
mtype = "UNSTRUCT"  # Always unstructured
xyz = np.array([model.nodes[nid].get_position() / 1000 for nid in model.nodes.keys()], dtype=np.float32,)

conec_list = [model.elements[eid].node_ids for eid in model.elements.keys()]
max_nodes = max(len(elem) for elem in conec_list)
conec = np.array([elem + [-1] * (max_nodes - len(elem)) for elem in conec_list], dtype=np.int32)

eltype = np.array([BDF2ELTYPE.get(model.elements[eid].type) for eid in model.elements.keys()], dtype=np.int32,)
cellO = np.arange(len(model.elements), dtype=np.int32)
pointO = np.array(list(model.nodes.keys()), dtype=np.int32)

# Create a mesh
mesh = pyLOM.Mesh(mtype, xyz, conec, eltype, cellO, pointO, None)
mesh.partition_table = pyLOM.PartitionTable.new(1, mesh.ncells, mesh.npoints)

with open(OUTFILE, "w") as f:
    mesh.save(OUTFILE)  # Store the mesh

pyLOM.cr_info()
