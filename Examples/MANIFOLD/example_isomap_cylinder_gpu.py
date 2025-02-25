#!/usr/bin/env python
#
# Isomap analysis (GPU implementation).
#
# Last revision: 11/02/2025
import pyLOM
import matplotlib.pyplot as plt

pyLOM.gpu_device()


## Parameters
DATAFILE = './DATA/CYLINDER.h5'
VARIABLE = 'VELOC'


## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table).to_gpu([VARIABLE])
X = d[VARIABLE]
t = d.get_variable('time')

Y,R,_ = pyLOM.MANIFOLD.isomap(X,2,12)

plt.scatter(-Y[0,:],-Y[1,:],s=15,c='b')

pyLOM.cr_info()
plt.show()
