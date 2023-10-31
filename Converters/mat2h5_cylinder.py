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
MATFILE = './CYLINDER_ALL.mat'
OUTFILE = './DATA/CYLINDER.h5'
DT      = 0.2
DIMSX   = -1., 8.
DIMSY   = -2., 2.


## Load MAT file
mat = loadmat(MATFILE)


## Create the mesh
nx, ny = int(mat['ny'])+1,int(mat['nx'])+1 # wrong in the mat file
mesh = pyLOM.Mesh.new_struct2D(nx,ny,None,None,DIMSX,DIMSY) 
print(mesh)


## Create partition table
ptable = pyLOM.PartitionTable.new(1,mesh.ncells,mesh.npoints)
print(ptable)

# Build time instants
time = DT*np.arange(mat['UALL'].shape[1])

# Build velocity as a 2D array
VELOC = np.zeros((2*mesh.ncells,time.shape[0]),dtype=np.double)
VELOC[:2*mesh.ncells:2,:]  = np.ascontiguousarray(mat['UALL'].astype(np.double))
VELOC[1:2*mesh.ncells:2,:] = np.ascontiguousarray(mat['VALL'].astype(np.double))

VORTI = np.zeros((1*mesh.ncells,time.shape[0]),dtype=np.double)
VORTI[:,:] = np.ascontiguousarray(mat['VORTALL'].astype(np.double))

VELOX = np.zeros((1*mesh.ncells,time.shape[0]),dtype=np.double)
VELOX[:,:] = np.ascontiguousarray(mat['UALL'].astype(np.double))


## Create dataset for pyLOM
d = pyLOM.Dataset(ptable=ptable, mesh=mesh, time=time,
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
