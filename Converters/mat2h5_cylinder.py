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
MATFILE = './Examples/Data/CYLINDER_ALL.mat'
OUTFILE = './Examples/Data/CYLINDER.h5'
DT      = 0.2
DIMSX   = -1., 8.
DIMSY   = -2., 2.


## Load MAT file
mat = loadmat(MATFILE)


## Build mesh information dictionary
mesh = {'type':'struct2D','nx':int(mat['ny'])+1,'ny':int(mat['nx'])+1} # wrong in the mat file

# Build node positions
dx = (DIMSX[1]-DIMSX[0])/(mesh['nx']-1.)
dy = (DIMSY[1]-DIMSY[0])/(mesh['ny']-1.)
x  = dx*np.arange(mesh['nx']) + DIMSX[0]
y  = dy*np.arange(mesh['ny']) + DIMSY[0]
xx, yy = np.meshgrid(x,y,indexing='ij')
xy = np.zeros((mesh['nx']*mesh['ny'],3),dtype=np.double)
xy[:,0] = xx.reshape((mesh['nx']*mesh['ny'],),order='C')
xy[:,1] = yy.reshape((mesh['nx']*mesh['ny'],),order='C')

# Build time instants
time = DT*np.arange(mat['UALL'].shape[1])

# Build velocity as a 2D array
nnx, nny = mesh['nx']-1,mesh['ny']-1
VELOC = np.zeros((2*nnx*nny,time.shape[0]),dtype=np.double)
VELOC[:2*nnx*nny:2,:]  = np.ascontiguousarray(mat['UALL'].astype(np.double))
VELOC[1:2*nnx*nny:2,:] = np.ascontiguousarray(mat['VALL'].astype(np.double))

VELOX = np.zeros((1*nnx*nny,time.shape[0]),dtype=np.double)
VELOX[:,:]  = np.ascontiguousarray(mat['UALL'].astype(np.double))

VORTI = np.zeros((1*nnx*nny,time.shape[0]),dtype=np.double)
VORTI[:,:] = np.ascontiguousarray(mat['VORTALL'].astype(np.double))


## Create dataset for pyLOM
d = pyLOM.Dataset(mesh=mesh, xyz=xy, time=time,
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	VELOC = {'point':False,'ndim':2,'value':VELOC},
	VELOX = {'point':False,'ndim':1,'value':VELOX},
	VORTI = {'point':False,'ndim':1,'value':VORTI},
)
print(d)

# Store dataset
d.save(OUTFILE)

pyLOM.cr_info()
