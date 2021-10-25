#!/usr/bin/env python
#
# Conversor of MAT file to HDF5 file.
#
# Last revision: 03/08/2021
from __future__ import print_function, division

import numpy as np, h5py
import pyLOM


## Parameters
MATFILE = './DATA/Tensor_re280.mat'
OUTFILE = './DATA/Tensor_re280.h5'
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
mesh = {'type':'struct3D','nx':nx,'ny':ny, 'nz':nz}

# Build node positions
x = np.linspace(DIMSX[0], DIMSX[1], mesh['nx'])
y = np.linspace(DIMSY[0], DIMSY[1], mesh['ny'])
z = np.linspace(DIMSZ[0], DIMSZ[1], mesh['nz'])
xx, yy, zz = np.meshgrid(x,y,z,indexing='ij')
xyz = np.zeros((mesh['nx']*mesh['ny']*mesh['nz'],3),dtype=np.double)
xyz[:,0] = xx.reshape((mesh['nx']*mesh['ny']*mesh['nz'],),order='C')
xyz[:,1] = yy.reshape((mesh['nx']*mesh['ny']*mesh['nz'],),order='C')
xyz[:,2] = zz.reshape((mesh['nx']*mesh['ny']*mesh['nz'],),order='C')

# Build time instants
time = DT*np.arange(nt) + DT
time = time[280:400]
nt   = time.shape[0]

# Obtain 3D velocity field
npoints = nx*ny*nz
VELOC = np.zeros((3*nx*ny*nz,nt),dtype=np.double)
VELOC[:npoints,:]             = tensor[0,:,:,:,280:400].reshape((nx*ny*nz,nt),order='C')
VELOC[npoints:2*npoints,:]    = tensor[1,:,:,:,280:400].reshape((nx*ny*nz,nt),order='C')
VELOC[2*nx*ny*nz:3*npoints,:] = tensor[2,:,:,:,280:400].reshape((nx*ny*nz,nt),order='C')


## Create dataset for pyLOM
d = pyLOM.Dataset(mesh=mesh, xyz=xyz, time=time,
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	VELOC = {'point':True,'ndim':3,'value':VELOC},
)
print(d)

# Store dataset
d.save(OUTFILE)

pyLOM.cr_info()
