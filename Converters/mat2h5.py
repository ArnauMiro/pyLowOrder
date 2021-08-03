#!/usr/bin/env python
#
# Conversor of MAT file to HDF5 file.
#
# Last revision: 03/08/2021
from __future__ import print_function, division

import numpy as np
from scipy.io import loadmat

import pyLOM


## Parameters
MATFILE = './DATA/CYLINDER_ALL.mat'
OUTFILE = './DATA/CYLINDER.h5'
DT      = 0.2
DIMSX   = -1., 8.
DIMSY   = -2., 2.

## Load MAT file
mat = loadmat(MATFILE) 


## Build mesh information dictionary
mesh = {'type':'struct2D','nx':int(mat['nx']),'ny':int(mat['ny'])}

# Build node positions
dx = (DIMSX[1]-DIMSX[0])/(mesh['nx']-1.)
dy = (DIMSY[1]-DIMSY[0])/(mesh['ny']-1.)
x  = dx*np.arange(mesh['nx']) + DIMSX[0]
y  = dy*np.arange(mesh['ny']) + DIMSY[0]
xx, yy = np.meshgrid(x,y)
xyz = np.zeros((mesh['nx']*mesh['ny'],2),dtype=np.double)
xyz[:,0] = xx.reshape((mesh['nx']*mesh['ny'],))
xyz[:,1] = yy.reshape((mesh['nx']*mesh['ny'],))

# Build time instants
time = DT*np.arange(mat['UALL'].shape[1])


## Create dataset for pyLOM
d = pyLOM.Dataset(mesh=mesh, xyz=xyz, time=time,
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	UALL   = np.ascontiguousarray(mat['UALL'].astype(np.double)),
	UEXTRA = np.ascontiguousarray(mat['UEXTRA'].astype(np.double)),
	VALL   = np.ascontiguousarray(mat['VALL'].astype(np.double)),
	VEXTRA = np.ascontiguousarray(mat['VEXTRA'].astype(np.double)),
)
print(d)

# Store dataset
d.save(OUTFILE)