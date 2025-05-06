#!/usr/bin/env python
#
# Example to showcase mesh utilities.
#
# Last revision: 01/02/2023
from __future__ import print_function, division

import numpy as np
import pyLOM


## Load cylinder mesh
DATAFILE = './Testsuite/DATA/CYLINDER.h5'

m = pyLOM.Mesh.load(DATAFILE)
print(m)


## Compute geometrical information (this may take a while depending on your mesh size)
xyz_center = m.xyzc # Cell nodes
surf_norms = m.normal
edge_norms = m.wall_normal
cell_conec = m.cell_connectivity

pyLOM.pprint(0,"surf_norms: ", surf_norms.shape)
pyLOM.pprint(0,"edge_norms: ", edge_norms.shape)
pyLOM.pprint(0,"cell_connectivity: ", cell_conec.shape)


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
	CELL_CONEC = {'ndim':cell_conec.shape[1],'value':cell_conec.flatten()},
)
pyLOM.io.pv_writer(m,d,'mesh',basedir='./',instants=[0],times=[0.],vars=['SURF_NORMS','EDGE_NORM_1', 'EDGE_NORM_2', 'EDGE_NORM_3', 'EDGE_NORM_4','CELL_CONEC'],fmt='vtkh5')


pyLOM.cr_info()