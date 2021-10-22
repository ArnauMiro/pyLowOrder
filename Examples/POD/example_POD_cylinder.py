#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import numpy as np

import pyLOM

## Parameters
DATAFILE = './DATA/CYLINDER.h5'
VARIABLE = 'VELOC'


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
X  = d[VARIABLE]
t  = d.time
dt = d.time[1] - d.time[0]


## Compute POD
pyLOM.cr_start('example',0)
# Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
pyLOM.plotResidual(S)
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=5e-6)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.POD.RMSE(X_POD,X)
pyLOM.cr_stop('example',0)

print('RMSE = %e'%rmse)


## Plot POD mode
# 0 - module, 1,2 - components
pyLOM.plotMode(PSI,d.xyz,V,t,d.mesh,d.info(VARIABLE),dim=0,modes=[1,2,3,4])


## Plot reconstructed flow
#pyLOM.plotSnapshot(X_POD[:,10],d.xyz,d.mesh,d.info('VELOC'))
fig,ax,anim = pyLOM.animateFlow(X,X_POD,d.xyz,d.mesh,d.info(VARIABLE),dim=0)


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()
