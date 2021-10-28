#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np

import pyLOM

## Parameters
DATAFILE = './DATA/Tensor_re280.h5'
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
print('RMSE = %.2e'%rmse)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_U',d.info(VARIABLE)['point'],6,0,d.extract_modes(PSI,1,modes=[1,4,6,2,5,3],point=d.info(VARIABLE)['point']))
d.add_variable('spatial_modes_V',d.info(VARIABLE)['point'],6,0,d.extract_modes(PSI,2,modes=[1,4,6,2,5,3],point=d.info(VARIABLE)['point']))
d.add_variable('spatial_modes_W',d.info(VARIABLE)['point'],6,0,d.extract_modes(PSI,3,modes=[1,4,6,2,5,3],point=d.info(VARIABLE)['point']))
d.write('modes',basedir='out',instants=[0],vars=['spatial_modes_U','spatial_modes_V','spatial_modes_W'],fmt='ensi')
pyLOM.io.Ensight_writeCase(os.path.join('out','modes.ensi.case'),'modes.ensi.geo',
	[
		{'name':'spatial_modes_U','dims':6,'point':d.info(VARIABLE)['point'],'file':'modes.ensi.spatial_modes_U-******'},
		{'name':'spatial_modes_V','dims':6,'point':d.info(VARIABLE)['point'],'file':'modes.ensi.spatial_modes_V-******'},
		{'name':'spatial_modes_W','dims':6,'point':d.info(VARIABLE)['point'],'file':'modes.ensi.spatial_modes_W-******'},
	],
	np.array([0.],np.double)
)

# Temporal evolution
d.add_variable('VELOR',d.info(VARIABLE)['point'],3,t.shape[0],X_POD)
d.write('flow',basedir='out',instants=np.arange(t.shape[0],dtype=np.int32),vars=['VELOC','VELOR'],fmt='ensi')
pyLOM.io.Ensight_writeCase(os.path.join('out','flow.ensi.case'),'flow.ensi.geo',
	[
		{'name':'VELOC','dims':3,'point':d.info(VARIABLE)['point'],'file':'flow.ensi.VELOC-******'},
		{'name':'VELOR','dims':3,'point':d.info(VARIABLE)['point'],'file':'flow.ensi.VELOR-******'},
	],
	t
)


## Plot POD mode
_,ax,_ = pyLOM.POD.plotMode(PSI,d.xyz,V,t,d.mesh,d.info(VARIABLE),modes=[1,3])
ax[0][2].set_xlim([0,0.5])
ax[1][2].set_xlim([0,0.5])


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()