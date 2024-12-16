#!/usr/bin/env python
#
# Conversor of MAT file to HDF5 file.
#
# Last revision: 03/08/2021
from __future__ import print_function, division

import numpy as np, h5py
import pyLOM


## Parameters
MATFILE = '../DATA/Tensor_re280.mat'
OUTFILE = './DATA/TENSOR280.h5'
DT      = 1.
DIMSX   = 0., 10.
DIMSY   = -2., 2.
DIMSZ   = 0., 1.67


## Load MAT file
f   = h5py.File(MATFILE,'r')
mat = { k:np.array(v) for k,v in f.items() }
f.close()

tensor = np.ascontiguousarray(np.transpose(mat['Tensor'],(4,3,2,1,0)))
[ndim,nx,ny,nz,nt] = tensor.shape


## Build mesh information dictionary
mesh = pyLOM.Mesh.new_struct3D(nx+1,ny+1,nz+1,None,None,None,DIMSX,DIMSY,DIMSZ) 
print(mesh)


## Create partition table
ptable = pyLOM.PartitionTable.new(1,mesh.ncells,mesh.npoints)
mesh.partition_table = ptable
print(ptable)


## Build time instants
time = DT*np.arange(nt) + DT
time = time[280:400]
nt   = time.shape[0]


## Obtain 3D velocity field
npoints = nx*ny*nz
VELOX = tensor[0,:,:,:,280:400].reshape((nx*ny*nz,nt),order='C')
VELOY = tensor[1,:,:,:,280:400].reshape((nx*ny*nz,nt),order='C')
VELOZ = tensor[2,:,:,:,280:400].reshape((nx*ny*nz,nt),order='C')


## Create dataset for pyLOM
nsample = 10
d = pyLOM.Dataset(xyz=mesh.xyzc, ptable=ptable, order=mesh.cellOrder, point=False,
	# Add the time as the only variable
	vars  = {'time':{'idim':0,'value':time[::nsample]}},
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	VELOX = {'ndim':1,'value':VELOX[:,::nsample].copy()},
)
print(d)

mesh.save(OUTFILE) # Store the mesh
d.save(OUTFILE)    # Store dataset


pyLOM.cr_info()