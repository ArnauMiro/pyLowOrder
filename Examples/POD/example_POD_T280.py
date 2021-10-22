#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import numpy as np

import pyLOM

## Parameters
DATAFILE = './DATA/Tensor_re280.h5'


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
X  = d['T']
t  = d.time
dt = d.time[1] - d.time[0]


## Compute POD
pyLOM.cr_start('example',0)
# Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=True) # PSI are POD modes
pyLOM.plotResidual(S)
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=5e-6)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.POD.RMSE(X_POD,X)
pyLOM.cr_stop('example',0)

print('RMSE = %.2e'%rmse)


## Plot POD mode
_,ax,_ = pyLOM.plotMode(PSI,d.xyz,V,t,d.mesh,modes=[1,3])
ax[2].set_xlim([0,0.5])


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()
