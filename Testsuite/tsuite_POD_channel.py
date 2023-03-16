#!/usr/bin/env python
#
# PYLOM Testsuite
# Run POD on the channel dataset
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np
import pyLOM


## Parameters
DATAFILE  = './CHANNEL.h5'
VARIABLES = ['VELOX','VELOY','VELOZ']


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
X = d.X(*VARIABLES)
t = d.time


## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
if pyLOM.utils.is_rank_or_serial(root=0): 
	fig,_ = pyLOM.POD.plotResidual(S)
	os.makedirs('channelPOD',exist_ok=True)
	fig.savefig('channelPOD/residuals.png',dpi=300)
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=3e-2)
pyLOM.POD.save('channelPOD/results.h5',PSI,S,V,d.partition_table,nvars=3,pointData=True)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)


## Dump to ParaView
# Spatial modes
modes = np.arange(1,5+1,dtype=np.int32)
d.add_variable('spatial_modes_U',True,len(modes),pyLOM.POD.extract_modes(PSI,1,d.mesh.npoints,modes=modes))
d.add_variable('spatial_modes_V',True,len(modes),pyLOM.POD.extract_modes(PSI,2,d.mesh.npoints,modes=modes))
d.add_variable('spatial_modes_W',True,len(modes),pyLOM.POD.extract_modes(PSI,3,d.mesh.npoints,modes=modes))
d.write('modes',basedir='channelPOD/modes',instants=[0],times=[0.],vars=['spatial_modes_U','spatial_modes_V','spatial_modes_W'],fmt='vtkh5')

# Temporal evolution
idx = 5
d2 = pyLOM.Dataset(ptable=d.partition_table, mesh=d.mesh, time=d.time[-idx:])
d2.add_variable('VELOC',True,3,X[:,-idx:])
d2.add_variable('VELOR',True,3,X_POD[:,-idx:])
d2.write('flow',basedir='channelPOD/flow',instants=np.arange(d2.time.shape[0],dtype=np.int32),times=d2.time,vars=['VELOC','VELOR'],fmt='vtkh5')


## Plot POD mode
if pyLOM.utils.is_rank_or_serial(0):
	# 0 - module, 1,2 - components
	fig, _ = pyLOM.POD.plotMode(V[:,:-1],t[:-1],modes=modes)
	for i,f in enumerate(fig): f.savefig('channelPOD/modes_%d.png'%i,dpi=300)


## Show and print timings
pyLOM.cr_info()
