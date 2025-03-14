#!/usr/bin/env python
#
# MDS analysis.
#
# Last revision: 11/02/2025
import matplotlib.pyplot as plt
import pyLOM

pyLOM.gpu_device(gpu_per_node=4)


## Parameters
DATAFILE = './DATA/CYLINDER.h5'
VARIABLE = 'VELOC'

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table).to_gpu([VARIABLE])
X = d[VARIABLE]
t = d.get_variable('time')

Y = pyLOM.MANIFOLD.mds(X,2)

plt.scatter(Y[0,:],Y[1,:],s=10,c='b')

pyLOM.cr_info()
plt.show()
