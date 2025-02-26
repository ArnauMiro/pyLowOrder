#!/usr/bin/env python
#
# Isomap analysis.
#
# Last revision: 11/02/2025
import os, numpy as np
import pyLOM
import matplotlib.pyplot as plt

## Parameters
DATAFILE = './DATA/CYLINDER.h5'
VARIABLE = 'VELOC'

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d[VARIABLE]
t = d.get_variable('time')

Y = pyLOM.MANIFOLD.mds(X,2)

plt.scatter(Y[0,:],Y[1,:],s=10,c='b')

pyLOM.cr_info()
plt.show()
