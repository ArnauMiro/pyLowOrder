#!/usr/bin/env python
#
# Isomap analysis.
#
# Last revision: 11/02/2025
import os, numpy as np
import pyLOM
import matplotlib.pyplot as plt

## Parameters
DATAFILE = './Testsuite/DATA/CYLINDER.h5'
VARIABLE = 'VELOC'

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d[VARIABLE]
t = d.get_variable('time')

Y,R,_ = pyLOM.MANIFOLD.isomap(X.T,2,12)

plt.scatter(-Y[0,:],-Y[1,:],s=15,c='b')
plt.show()