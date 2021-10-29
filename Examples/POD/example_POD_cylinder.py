#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np

import pyLOM

## Parameters
DATAFILE = './DATA/CYLINDER.h5'
VARIABLE = 'VELOC'


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
X  = d[VARIABLE]
t  = d.time


## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
pyLOM.POD.plotResidual(S)
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=5e-6)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
print('RMSE = %e'%rmse)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_U',False,6,0,pyLOM.POD.extract_modes(PSI,1,d.mesh,modes=[1,4,6,2,5,3],point=d.info(VARIABLE)['point']))
d.add_variable('spatial_modes_V',False,6,0,pyLOM.POD.extract_modes(PSI,2,d.mesh,modes=[1,4,6,2,5,3],point=d.info(VARIABLE)['point']))
d.write('modes',basedir='out',instants=[0],vars=['spatial_modes_U','spatial_modes_V'],fmt='ensi')
pyLOM.io.Ensight_writeCase(os.path.join('out','modes.ensi.case'),'modes.ensi.geo',
	[
		{'name':'spatial_modes_U','dims':6,'point':d.info(VARIABLE)['point'],'file':'modes.ensi.spatial_modes_U-******'},
		{'name':'spatial_modes_V','dims':6,'point':d.info(VARIABLE)['point'],'file':'modes.ensi.spatial_modes_V-******'},
	],
	np.array([0.],np.double)
)

# Temporal evolution
d.add_variable('VELOR',False,2,t.shape[0],X_POD)
d.write('flow',basedir='out',instants=np.arange(t.shape[0],dtype=np.int32),vars=['VELOC','VELOR'],fmt='ensi')
pyLOM.io.Ensight_writeCase(os.path.join('out','flow.ensi.case'),'flow.ensi.geo',
	[
		{'name':'VELOC','dims':2,'point':d.info(VARIABLE)['point'],'file':'flow.ensi.VELOC-******'},
		{'name':'VELOR','dims':2,'point':d.info(VARIABLE)['point'],'file':'flow.ensi.VELOR-******'},
	],
	t
)


## Plots
# POD mode: 0 - module, 1,2 - components
pyLOM.POD.plotMode(PSI,d.xyz,V,t,d.mesh,d.info(VARIABLE),dim=0,modes=[1,2,3,4])

# Plot reconstructed flow
#pyLOM.POD.plotSnapshot(X_POD[:,10],d.xyz,d.mesh,d.info(VARIABLE))
fig,ax,anim = pyLOM.POD.animateFlow(X,X_POD,d.xyz,d.mesh,d.info(VARIABLE),dim=0)


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()
