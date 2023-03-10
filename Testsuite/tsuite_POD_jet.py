#!/usr/bin/env python
#
# PYLOM Testsuite
# Run POD on the jet dataset
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np
import pyLOM


## Parameters
DATAFILE = './JET.h5'
VARIABLE = 'PRESS'


## Data loadingx
d = pyLOM.Dataset.load(DATAFILE)
X  = d[VARIABLE]
t  = d.time


## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
if pyLOM.utils.is_rank_or_serial(root=0): 
	fig,_ = pyLOM.POD.plotResidual(S)
	os.makedirs('jetPOD',exist_ok=True)
	fig.savefig('jetPOD/residuals.png',dpi=300)
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=5e-6)
pyLOM.POD.save('jetPOD/results.h5',PSI,S,V,d.partition_table,nvars=1,pointData=True)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_P',True,6,pyLOM.POD.extract_modes(PSI,1,d.mesh.npoints,modes=[1,4,6,2,5,3]))
d.write('modes',basedir='jetPOD/modes',instants=[0],times=[0.],vars=['spatial_modes_P'],fmt='vtkh5')
#for imode in [0,1,2,3,4]:
#	pyLOM.POD.plotSnapshot(d,vars=['spatial_modes_P'],instant=0,component=imode,cmap='jet',cpos='xy',off_screen=True,screenshot='jetPOD/mode_P_%d_%d.png'%(pyLOM.utils.MPI_RANK,imode))

# Temporal evolution
d.add_variable('PRESR',True,1,X_POD)
d.write('flow',basedir='jetPOD/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['PRESS','PRESR'],fmt='vtkh5')
#pyLOM.POD.plotSnapshot(d,vars=['PRESR'],instant=0,component=0,cmap='jet',cpos='xy',off_screen=True,screenshot='jetPOD/P_%d.png'%pyLOM.utils.MPI_RANK)


## Plot POD mode
if pyLOM.utils.is_rank_or_serial(0):
	# 0 - module, 1,2 - components
	fig, _ = pyLOM.POD.plotMode(V[:,:-1],t[:-1],modes=[1,2,3,4],scale_freq=2.56)
	for i,f in enumerate(fig): f.savefig('jetPOD/modes_%d.png'%i,dpi=300)


## Show and print timings
pyLOM.cr_info()