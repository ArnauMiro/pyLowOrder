#!/usr/bin/env python
#
# Isomap analysis.
#
# Last revision: 11/02/2025

import os, numpy as np
import pyLOM
from pyLOM.MANIFOLD.wrapper import isomap
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap

## Parameters
DATAFILE = './Testsuite/DATA/CYLINDER.h5'
VARIABLE = 'VELOC'


## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d[VARIABLE]
t = d.get_variable('time')

Y,R,_ = isomap(X.T,2,12)

Comp = Isomap(n_neighbors=12,
                neighbors_algorithm='brute',
                metric="minkowski",
                p=2,
                metric_params=None,
                n_components=2).fit_transform(X.T)

print(Y)

print(R)

plt.scatter(-Y[0,:],-Y[1,:],s=15,c='b')
plt.scatter(Comp[:,0],Comp[:,1],s=10,c='r')
plt.show()