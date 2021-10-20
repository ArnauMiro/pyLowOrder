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
mesh = {'type':'struct2D','nx':int(mat['nx']),'ny':int(mat['nr'])}
p = mat['p']
p_reshaped1 = p.reshape((mesh['nx']*mesh['ny'], 5000))

# Build node positions
xx = mat['x']
yy = mat['r']
xyz = np.zeros((mesh['nx']*mesh['ny'],2),dtype=np.double)
xyz[:,0] = xx.reshape((mesh['nx']*mesh['ny'],), order = 'C')
xyz[:,1] = yy.reshape((mesh['nx']*mesh['ny'],), order = 'C')

# Build time instants
time = DT*np.arange(p.shape[2])+0.04


## Create dataset for pyLOM
d = pyLOM.Dataset(mesh=mesh, xyz=xyz, time=time,
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	p   = np.ascontiguousarray(p_reshaped1.astype(np.double)),
)
print(d)

# Store dataset
d.save(OUTFILE)

pyLOM.cr_info()
