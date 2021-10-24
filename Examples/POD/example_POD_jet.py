#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np

import pyLOM

## Parameters
DATAFILE = './DATA/jetLES.h5'
VARIABLE = 'PRESS'

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

print('RMSE = %.2e'%rmse)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_P',True,6,0,d.extract_modes(PSI,1,modes=[1,4,6,2,5,3],point=d.info(VARIABLE)['point']))
d.write('modes',basedir='out',instants=[0],vars=['spatial_modes_P'],fmt='ensi')
pyLOM.io.Ensight_writeCase(os.path.join('out','modes.ensi.case'),'modes.ensi.geo',
	[
		{'name':'spatial_modes_P','dims':6,'point':d.info(VARIABLE)['point'],'file':'modes.ensi.spatial_modes_P-******'},
	],
	np.array([0.],np.double)
)

# Temporal evolution
d.add_variable('PRESR',True,1,t.shape[0],X_POD)
d.write('flow',basedir='out',instants=np.arange(t.shape[0],dtype=np.int32),vars=['PRESS','PRESR'],fmt='ensi')
pyLOM.io.Ensight_writeCase(os.path.join('out','flow.ensi.case'),'flow.ensi.geo',
	[
		{'name':'PRESS','dims':1,'point':d.info(VARIABLE)['point'],'file':'flow.ensi.PRESS-******'},
		{'name':'PRESR','dims':1,'point':d.info(VARIABLE)['point'],'file':'flow.ensi.PRESR-******'},
	],
	t
)


## Plot POD mode
pyLOM.plotMode(PSI,d.xyz,V,t,d.mesh,d.info(VARIABLE),modes=[1,2,3,4],scale_freq=2.56)
#pyLOM.plotSnapshot(X_POD[:,10],d.xyz,d.mesh)
fig,ax,anim = pyLOM.animateFlow(X,X_POD,d.xyz,d.mesh,d.info(VARIABLE))


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()
