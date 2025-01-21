#!/usr/bin/env python
#
# Example of how to run a POD on runtime.
#
# Last revision: 20/01/2025
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np
import pyLOM

## Parameters
DATAFILE = './DATA/CYLINDER.h5'
VARIABLE = 'VELOC'

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
U = d[VARIABLE]
t = d.get_variable('time')

## Set up parameters
nbatches  = 4
nt        = t.shape[0]
batchsize = int(np.floor(nt/nbatches)) 
nmod      = 4
q         = 1

## Initialize the QB with the first batch
X        = U[:,:batchsize]
Q1,B1,Yo = pyLOM.math.init_qr_streaming(X, nmod, q)
Xr       = pyLOM.math.matmul(Q1,B1)
Ek       = pyLOM.math.energy(Xr,X)
pyLOM.pprint(0, 'Partial energy recovered', Ek, flush=True)

## Iterate over the rest of batches
for ii in range(nbatches-1):
	X        = U[:,(ii+1)*batchsize:(ii+2)*batchsize]
	Q1,B1,Yo = pyLOM.math.update_qr_streaming(X,Q1,B1,Yo,nmod,q)
	Xr       = pyLOM.math.matmul(Q1,B1[:,-batchsize:].copy())
	Ek       = pyLOM.math.energy(Xr,X)
	pyLOM.pprint(0, ii, 'Partial energy recovered', Ek, flush=True)