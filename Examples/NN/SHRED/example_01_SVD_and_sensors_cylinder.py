#!/usr/bin/env python
#
# Example of POD to reduce dimensionality for SHRED architecture.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np
import pyLOM

## Parameters
DATAFILE = '/gpfs/scratch/bsc21/bsc021828/DATA_PYLOM/CYLINDER.h5'
VARLIST  = ['VELOX', 'VORTI']

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
t = d.get_variable('time')
N = t.shape[0]

## Divide in training, validation and test
tridx, vaidx, teidx = pyLOM.math.data_splitting(N, mode='reconstruct')

## Extract sensors
# Generate random sensors
nsens  = 20     # Number of sensors
x0, x1 = 0.5, 8 # Bounds at the X axis of the region where the sensor will be located
y0, y1 = -1, 1  # Bounds at the Y axis of the region where the sensor will be located
bounds = np.array([x0,x1,y0,y1])
dsens  = d.select_random_sensors(nsens, bounds, VARLIST)
## Split the sensors in training, validation and test
dstrai = dsens.mask_fields(tridx,'time')
dsvali = dsens.mask_fields(vaidx,'time')
dstest = dsens.mask_fields(teidx,'time')
# Save the sensor dataset
dstrai.save('sensors_trai.h5', nopartition=True)
dsvali.save('sensors_vali.h5', nopartition=True)
dstest.save('sensors_test.h5', nopartition=True)

## Separate the full space datasets to compute POD
dtrai = d.mask_fields(tridx,'time')
dvali = d.mask_fields(vaidx,'time')
dtest = d.mask_fields(teidx,'time')
## Compute POD separately for each variable in order to reduce memory usage during the SVD. POD is computed only for the training dataset. Validation and test are projected to the POD modes
for var in VARLIST:
    ## Fetch training dataset and compute POD
    Xtrai   = dtrai[var]
    PSI,S,V = pyLOM.POD.run(Xtrai,remove_mean=True,randomized=True,r=8,q=3)
    ## Save POD modes for training of each variable
    pyLOM.POD.save('POD_trai_%s.h5'%var,dtrai.partition_table,U=PSI,S=S,V=V,nvars=1,pointData=dtrai.point)
    ## Fetch validation dataset and project POD modes
    Xvali   = dvali[var]
    proj    = pyLOM.math.matmulp(PSI.T, Xvali)
    Vvali   = pyLOM.math.matmul(pyLOM.math.diag(1/S), proj)
    ## Save POD projection of validation data of each variable
    pyLOM.POD.save('POD_vali_%s.h5'%var,dvali.partition_table,V=Vvali,nvars=1,pointData=dvali.point)
    ## Fetch test dataset and project POD modes
    Xtest   = dtest[var]
    proj    = pyLOM.math.matmulp(PSI.T, Xtest)
    Vtest   = pyLOM.math.matmul(pyLOM.math.diag(1/S), proj)
    ## Save POD projection of validation data of each variable
    pyLOM.POD.save('POD_test_%s.h5'%var,dtest.partition_table,V=Vtest,nvars=1,pointData=dtest.point)     

## print timings
pyLOM.cr_info()