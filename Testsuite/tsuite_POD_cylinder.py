#!/usr/bin/env python
#
# PYLOM Testsuite
# Run POD on the cylinder dataset
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np
import pyLOM


## Parameters
DATAFILE = './CYLINDER.h5'
VARIABLE = 'VELOX'


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
X  = d[VARIABLE]
t  = d.time


## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
if pyLOM.utils.is_rank_or_serial(root=0): 
	fig,_ = pyLOM.POD.plotResidual(S)
	os.makedirs('cylinderPOD',exist_ok=True)
	fig.savefig('cylinderPOD/residuals.png',dpi=300)
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=5e-6)
pyLOM.POD.save('cylinderPOD/results.h5',PSI,S,V,d.partition_table,nvars=1,pointData=False)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_U',False,6,0,pyLOM.POD.extract_modes(PSI,1,d.mesh.ncells,modes=[1,4,6,2,5,3]))
d.write('modes',basedir='cylinderPOD/modes',instants=[0],times=[0.],vars=['spatial_modes_U'],fmt='vtkh5')
#for imode in [0,1,2,3,4]:
#	pyLOM.POD.plotSnapshot(d,vars=['spatial_modes_U'],instant=0,component=imode,cmap='jet',cpos='xy',off_screen=True,screenshot='cylinderPOD/mode_U_%d_%d.png'%(pyLOM.utils.MPI_RANK,imode))

# Temporal evolution
d.add_variable('VELXR',False,1,t.shape[0],X_POD)
d.write('flow',basedir='cylinderPOD/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['VELOX','VELXR'],fmt='vtkh5')
#pyLOM.POD.plotSnapshot(d,vars=['VELXR'],instant=0,component=0,cmap='jet',cpos='xy',off_screen=True,screenshot='cylinderPOD/U_%d.png'%pyLOM.utils.MPI_RANK)


## Plot POD mode
if pyLOM.utils.is_rank_or_serial(0):
	# 0 - module, 1,2 - components
	fig, _ = pyLOM.POD.plotMode(V[:,:-1],t[:-1],modes=[1,2,3,4])
	for i,f in enumerate(fig): f.savefig('cylinderPOD/modes_%d.png'%i,dpi=300)


## Show and print timings
pyLOM.cr_info()
