#!/usr/bin/env python
#
# Conversor of MAT file to HDF5 file.
#
# Last revision: 03/08/2021
from __future__ import print_function, division

import numpy as np, h5py
import pyLOM


## Parameters
MATFILE = './DATA/jetLES.mat'
OUTFILE = './DATA/jetLES.h5'


## Load MAT file
f   = h5py.File(MATFILE,'r')
mat = { k:np.array(v) for k,v in f.items() }
f.close()
DT  = float(mat['dt'])


## Build mesh information dictionary
mesh  = {'type':'struct2D','nx':int(mat['nx']),'ny':int(mat['nr'])}
PRESS = np.ascontiguousarray(mat['p'].reshape((mesh['nx']*mesh['ny'], 5000)).astype(np.double))

# Build node positions
x = np.unique(mat['x'])
y = np.unique(mat['r'])
xx, yy = np.meshgrid(x,y,indexing='ij')
xyz = np.zeros((mesh['nx']*mesh['ny'],3),dtype=np.double)
xyz[:,0] = xx.reshape((mesh['nx']*mesh['ny'],),order='C')
xyz[:,1] = yy.reshape((mesh['nx']*mesh['ny'],),order='C')

# Build time instants
time = DT*np.arange(mat['p'].shape[2])+0.04


## Create dataset for pyLOM
d = pyLOM.Dataset(mesh=mesh, xyz=xyz, time=time,
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	PRESS = {'point':True,'ndim':1,'value':PRESS},
)
print(d)

# Store dataset
d.save(OUTFILE)

pyLOM.cr_info()
