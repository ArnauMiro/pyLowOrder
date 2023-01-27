#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np
import pyLOM


## Parameters
DATAFILE = 'Examples/Data/jetLES.h5'
VARIABLE = 'PRESS'


## Data loadingx
d = pyLOM.Dataset.load(DATAFILE)
X  = d[VARIABLE][:,::10]
t  = d.time[::10]


## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
if pyLOM.utils.is_rank_or_serial(root=0): pyLOM.POD.plotResidual(S)
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=5e-6)
pyLOM.POD.save('results.h5',PSI,S,V,d.partition_table,nvars=1,pointData=True)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_P',True,6,0,pyLOM.POD.extract_modes(PSI,1,d.mesh.npoints,modes=[1,4,6,2,5,3]))
d.write('modes',basedir='out/modes',instants=[0],times=[0.],vars=['spatial_modes_P'],fmt='vtkh5')
pyLOM.POD.plotSnapshot(d,vars=['spatial_modes_P'],instant=0,component=0,cmap='jet',cpos='xy')

# Temporal evolution
d.add_variable('PRESR',True,1,t.shape[0],X_POD)
d.write('flow',basedir='out/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['PRESS','PRESR'],fmt='vtkh5')
pyLOM.POD.plotSnapshot(d,vars=['PRESR'],instant=0,component=0,cmap='jet',cpos='xy')


## Plot POD mode
if pyLOM.utils.is_rank_or_serial(0):
	# 0 - module, 1,2 - components
	pyLOM.POD.plotMode(V,t,modes=[1,2,3,4],scale_freq=2.56)


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()
