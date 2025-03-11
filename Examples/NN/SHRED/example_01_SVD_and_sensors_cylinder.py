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

## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False,randomized=True,r=8,q=3)

## Save POD to fit SHRED
pyLOM.POD.save('results.h5',PSI,S,V,d.partition_table,nvars=1,pointData=d.point)

## print timings
pyLOM.cr_info()