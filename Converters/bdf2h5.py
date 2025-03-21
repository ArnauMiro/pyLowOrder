import pyLOM
import numpy as np
from pyNastran.bdf.bdf import BDF
 
BDFFILE = ""
OUTFILE = ""
 
BDF2ELTYPE = {
    'CTRIA3': 2,  # Triangular cell
    'CQUAD4': 3,  # Quadrangular cell
}
 
# Load and read BDF file
model = BDF()
model.read_bdf(BDFFILE, xref=False)
 
# Create needed variables
mtype = 'UNSTRUCT' # Always unstructured
xyz = np.array([model.nodes[nid].get_position()/1000 for nid in model.nodes.keys()], dtype=np.float64)
 
connec_list = [model.elements[eid].node_ids for eid in model.elements.keys()]
connec_list = [[eid-1 for eid in row] for row in connec_list]
max_nodes = max(len(elem) for elem in connec_list)
connec = np.array([elem + [-1] * (max_nodes - len(elem)) for elem in connec_list], dtype=np.int32)
 
eltype = np.array([BDF2ELTYPE.get(model.elements[eid].type) for eid in model.elements.keys()], dtype=np.int32)
cellO = np.array(list(model.elements.keys()), dtype=np.int32) - 1
pointO = np.array(list(model.nodes.keys()), dtype=np.int32) - 1
 
# Create a mesh
mesh = pyLOM.Mesh(mtype,xyz,connec,eltype,cellO,pointO,None)
mesh.partition_table = pyLOM.PartitionTable.new(1,mesh.ncells,mesh.npoints)
 
mesh.save(OUTFILE,mode='w') # Store the mesh, overwrite
 
pyLOM.cr_info()