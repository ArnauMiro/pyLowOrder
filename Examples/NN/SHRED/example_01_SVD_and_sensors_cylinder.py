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
DATAFILE  = '/gpfs/scratch/bsc21/bsc021828/DATA_PYLOM/CYLINDER.h5'
DATAFIL2  = './CYLINDER.h5'
VARLIST  = ['VELOX', 'VORTI']


## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
t = d.get_variable('time')
N = t.shape[0]


## Divide in training, validation and test and append mask to current dataset
tridx, vaidx, teidx = d.split_data('time', mode='reconstruct')
pyLOM.pprint(0, tridx.shape, vaidx.shape, teidx.shape, t.shape)
# Regenerate output datafile with the variable masks
m.save(DATAFIL2, nopartition=True, mode='w')
d.save(DATAFIL2, nopartition=True)


## Extract sensors
# Generate random sensors
nsens  = 20     # Number of sensors
x0, x1 = 0.5, 8 # Bounds at the X axis of the region where the sensor will be located
y0, y1 = -1, 1  # Bounds at the Y axis of the region where the sensor will be located
bounds = np.array([x0,x1,y0,y1])
dsens  = d.select_random_sensors(nsens, bounds, VARLIST)
# Save the sensor dataset
dsens.save('sensors.h5', nopartition=True, mode='w')


## Compute POD separately for each variable in order to reduce memory usage during the SVD. POD is computed only for the training dataset. Validation and test are projected to the POD modes
for var in VARLIST:
    ## Fetch training dataset and compute POD
    Xtrai   = d.mask_field(var, tridx)
    PSI,S,V = pyLOM.POD.run(Xtrai,remove_mean=True,randomized=True,r=8,q=3)
    ## Save POD modes for training of each variable
    pyLOM.POD.save('POD_trai_%s.h5'%var,PSI,S,V,d.partition_table,nvars=1,pointData=d.point)
    ## Fetch validation dataset and project POD modes
    Xvali   = d.mask_field(var, vaidx)
    proj    = pyLOM.math.matmulp(PSI.T, Xvali)
    Vvali   = pyLOM.math.matmul(pyLOM.math.diag(1/S), proj)
    ## Save POD projection of validation data of each variable
    pyLOM.POD.save('POD_vali_%s.h5'%var,None,None,Vvali,d.partition_table,nvars=1,pointData=d.point)
    ## Fetch test dataset and project POD modes
    Xtest   = d.mask_field(var, teidx)
    proj    = pyLOM.math.matmulp(PSI.T, Xtest)
    Vtest   = pyLOM.math.matmul(pyLOM.math.diag(1/S), proj)
    ## Save POD projection of validation data of each variable
    pyLOM.POD.save('POD_test_%s.h5'%var,None,None,Vtest,d.partition_table,nvars=1,pointData=d.point)     


pyLOM.cr_info()