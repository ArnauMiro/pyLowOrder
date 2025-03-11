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
DATAFILE = '../../../Testsuite/DATA/CYLINDER.h5'
VARLIST  = ['VELOX', 'VORTI']

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)

## Extract sensors
# Generate random sensors
nsens  = 20     # Number of sensors
x0, x1 = 0.5, 8 # Bounds at the X axis of the region where the sensor will be located
y0, y1 = -1, 1  # Bounds at the Y axis of the region where the sensor will be located
bounds = np.array([x0,x1,y0,y1])
dsens  = d.select_random_sensors(nsens, bounds, VARLIST)
dsens.save('sensors.h5', nopartition=True)

## Run POD (separately for each variable in order to reduce memory usage during the SVD)
for var in VARLIST:
    X = d[var]
    PSI,S,V = pyLOM.POD.run(X,remove_mean=True,randomized=True,r=8,q=3)
    ## Save POD of each variable
    pyLOM.POD.save('POD_modes_%s.h5'%var,PSI,S,V,d.partition_table,nvars=1,pointData=d.point)

## print timings
pyLOM.cr_info()