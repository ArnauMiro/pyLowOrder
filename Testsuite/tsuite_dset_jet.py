#!/usr/bin/env python
#
# PYLOM Testsuite
# Build jet dataset for the testsuite
#
# Last revision: 03/08/2021
from __future__ import print_function, division

import numpy as np, h5py
import pyLOM


## Parameters
MATFILE = '../DATA/jetLES.mat'
OUTFILE = './JET.h5'


## Load MAT file
f    = h5py.File(MATFILE,'r')
mat  = { k:np.array(v) for k,v in f.items() }
f.close()
DT   = float(mat['dt'])
time = DT*np.arange(mat['p'].shape[2])+0.04


## Build mesh information dictionary
nx, ny = int(mat['nx']),int(mat['nr']) 
x = np.unique(mat['x'])
y = np.unique(mat['r'])
mesh = pyLOM.Mesh.new_struct2D(nx,ny,x,y,None,None) 
print(mesh)


## Obtain pressure array
PRESS = np.ascontiguousarray(mat['p'].reshape((mesh.npoints, 5000)).astype(np.double))


## Create partition table
ptable = pyLOM.PartitionTable.new(1,mesh.ncells,mesh.npoints)
print(ptable)


## Create dataset for pyLOM
nsample = 20
d = pyLOM.Dataset(ptable=ptable, mesh=mesh, time=time[::nsample],
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	PRESS = {'point':True,'ndim':1,'value':PRESS[:,::nsample]},
)
print(d)

# Store dataset
d.save(OUTFILE)

pyLOM.cr_info()
