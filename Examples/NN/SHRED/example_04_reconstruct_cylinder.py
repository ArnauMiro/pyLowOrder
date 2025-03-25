#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np
import pyLOM


## Parameters
DATAFILE = '/gpfs/scratch/bsc21/bsc021828/DATA_PYLOM/CYLINDER.h5'
VARIABLE = 'VELOX'


## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d[VARIABLE]
t = d.get_variable('time')


## Load POD spatial modes and its energy computed with the training snapshots
PSI, S = pyLOM.POD.load('POD_trai_%s.h5' % VARIABLE, vars=['U','S'])


## Load the predicted POD coefficients from SHRED
V = pyLOM.POD.load('POD_predicted_%s.h5' % VARIABLE, vars='V')[0]


## Reconstruct the flow and readd the temporal mean (SHRED was trained without)
X_POD  = pyLOM.POD.reconstruct(PSI,S,V)
mean   = pyLOM.math.temporal_mean(X)
X_PODm = pyLOM.math.subtract_mean(X_POD, -1*mean)


## Compute RMSE with original data
rmse = pyLOM.math.RMSE(X_PODm,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)


## Dump reconstruction to ParaView
d.add_field('VELOR',1,X_PODm)
pyLOM.io.pv_writer(m,d,'flow',basedir='out/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['VELOX','VELOR'],fmt='vtkh5')


pyLOM.cr_info()