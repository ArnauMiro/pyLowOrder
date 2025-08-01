#!/usr/bin/env python
#
# Example to showcase mesh utilities.
#
# Last revision: 01/02/2023
from __future__ import print_function, division

import numpy as np
import pyLOM
from pyLOM import pprint


## Load cylinder mesh
DATAFILE = './Testsuite/DATA/CYLINDER.h5'

m = pyLOM.Mesh.load(DATAFILE)
print(m)


## Compute geometrical information (this may take a while depending on your mesh size)
xyz_center = m.xyzc # Cell nodes
surf_norms = m.normal
edge_norms = m.wall_normal
cell_conec = m.cell_connectivity

# Ensure that the normals are coherent
edge_dict = pyLOM.vmmath.edge_to_cells(cell_conec)
surf_norms = pyLOM.vmmath.fix_normals_coherence(surf_norms, edge_dict, cell_conec, m.ncells)

# Create a dummy variable to test saving
dummy_var = np.zeros((2*m.ncells,3),np.double)

pyLOM.pprint(0,"surf_norms: ", surf_norms.shape)
pyLOM.pprint(0,"edge_norms: ", edge_norms.shape)
pyLOM.pprint(0,"cell_connectivity: ", cell_conec.shape)

# print("Partition table info:")
# print(m.partition_table)
# print("ptable split check:", m.partition_table.check_split())
# print("ptable n_partitions:", m.partition_table._nparts)
# print("ptable length:", len(m.partition_table))

## Export to ParaView for visualization
d = pyLOM.Dataset(xyz=xyz_center, ptable=m.partition_table, order=m.cellOrder, point=False,
	# Add the time as the only variable
	vars  = {'time':{'idim':0,'value':np.array([0])}},
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	SURF_NORMS = {'ndim':surf_norms.shape[1],'value':surf_norms.flatten()},
    EDGE_NORM_1 = {'ndim': 3,'value':edge_norms[:,:3].flatten()},
	EDGE_NORM_2 = {'ndim': 3,'value':edge_norms[:,3:6].flatten()},
	EDGE_NORM_3 = {'ndim': 3,'value':edge_norms[:,6:9].flatten()},
	EDGE_NORM_4 = {'ndim': 3,'value':edge_norms[:,9:12].flatten()},
	CELL_CONEC = {'ndim': 4,'value':cell_conec.flatten()},

	# SURF_NORMS = {'ndim': 3,'value':surf_norms.reshape(-1,3)},
    # EDGE_NORM_1 = {'ndim': 3,'value':edge_norms[:,:3].reshape(-1,3)},
	# EDGE_NORM_2 = {'ndim': 3,'value':edge_norms[:,3:6].reshape(-1,3)},
	# EDGE_NORM_3 = {'ndim': 3,'value':edge_norms[:,6:9].reshape(-1,3)},
	# EDGE_NORM_4 = {'ndim': 3,'value':edge_norms[:,9:12].reshape(-1,3)},
	# CELL_CONEC = {'ndim': 4,'value':cell_conec.reshape(-1,4)},

	# SURF_NORMS = {'ndim':surf_norms.shape[1],'value':surf_norms},
    # EDGE_NORM_1 = {'ndim': 3,'value':edge_norms[:,:3]},
	# EDGE_NORM_2 = {'ndim': 3,'value':edge_norms[:,3:6]},
	# EDGE_NORM_3 = {'ndim': 3,'value':edge_norms[:,6:9]},
	# EDGE_NORM_4 = {'ndim': 3,'value':edge_norms[:,9:12]},
	# CELL_CONEC = {'ndim': 4,'value':cell_conec},
	# Add the dummy variable
	# DUMMYVAR = {'ndim':dummy_var.shape[1],'value':dummy_var.flatten()},
)
print("SURF_NORMS:", surf_norms.shape, "→", surf_norms.flatten().shape)
print("ptable:", m.partition_table, "cells:", m.ncells)

pprint(0, "Dataset created with the following variables:")
print(d)
d.save('./example_mesh.h5')
pprint(0, "Dataset saved to ./example_mesh.h5")

pprint(0, "Loading the dataset back to verify")
d_load = pyLOM.Dataset.load('./example_mesh.h5')
pprint(d_load)
pyLOM.io.pv_writer(m,d,'mesh',basedir='./',instants=[0],times=[0.],vars=['SURF_NORMS','EDGE_NORM_1', 'EDGE_NORM_2', 'EDGE_NORM_3', 'EDGE_NORM_4','CELL_CONEC'],fmt='vtkh5')
pprint(0, "Mesh saved to ./mesh.vtkh5")
pyLOM.cr_info()