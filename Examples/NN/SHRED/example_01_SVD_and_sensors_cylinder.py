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
VARIABLE = 'VELOX'

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d[VARIABLE]
t = d.get_variable('time')

## Extract sensors
# Generate random sensors
nsens  = 15 # Number of sensors
x0, x1 = np.min(m.xyz[:,0]), np.max(m.xyz[:,0]) # Bounds at the X axis of the region where the sensor will be located
y0, y1 = np.min(m.xyz[:,1]), np.max(m.xyz[:,1]) # Bounds at the Y axis of the region where the sensor will be located
bounds = np.array([x0,x1,y0,y1])
dsens  = d.select_random_sensors(nsens, bounds)
dsens.save('sensors.h5')

## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=True,randomized=True,r=8,q=3)

## Save POD to fit SHRED
pyLOM.POD.save('POD_modes.h5',PSI,S,V,d.partition_table,nvars=1,pointData=d.point)

## print timings
pyLOM.cr_info()