#!/usr/bin/env python
#
# POD analysis.
#
# Last revision: 29/10/2021
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import os, numpy as np
import pyLOM


## Data loading
DATAFILE  = '../channel.h5'
VARIABLES = ['PRESS','VELOX','VELOY','VELOZ']

d = pyLOM.Dataset.load(DATAFILE)
X = d.X(*VARIABLES)

## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
if pyLOM.utils.is_rank_or_serial(root=0): 
	fig,_ = pyLOM.POD.plotResidual(S)
	fig.savefig('residuals.png',dpi=300)

# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=3e-2)

# Store results
pyLOM.POD.save('results.h5',PSI,S,V,d.partition_table,nvars=4,pointData=True)

# Spatial modes
modes = np.arange(1,100+1,dtype=np.int32)
d.add_variable('spatial_modes_P',True,len(modes),pyLOM.POD.extract_modes(PSI,1,d.mesh.npoints,modes=modes))
d.add_variable('spatial_modes_U',True,len(modes),pyLOM.POD.extract_modes(PSI,2,d.mesh.npoints,modes=modes))
d.add_variable('spatial_modes_V',True,len(modes),pyLOM.POD.extract_modes(PSI,3,d.mesh.npoints,modes=modes))
d.add_variable('spatial_modes_W',True,len(modes),pyLOM.POD.extract_modes(PSI,4,d.mesh.npoints,modes=modes))
d.write('modes',basedir='modes',instants=[0],times=[0.],vars=['spatial_modes_P','spatial_modes_U','spatial_modes_V','spatial_modes_W'],fmt='vtkh5')

# Plot POD modes
if pyLOM.utils.is_rank_or_serial(0):
	# 0 - module, 1,2 - components
	os.makedirs('modes',exist_ok=True)
	figs,_ = pyLOM.POD.plotMode(V[:,:-1],d.time[:-1],modes=[1,2,3,4])
	for ifig,fig in enumerate(figs): fig.savefig('modes/mode_%d.png'%ifig) 


## Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)

# Create a dataset to store the last 500 instants from the simulation
# and the reconstructed flow
idx     = 100
time    = d.time[-idx:]
X_PRESS = X[0:4*d.mesh.npoints:4,-idx:]
X_PRESR = X_POD[0:4*d.mesh.npoints:4,-idx:]
X_VELOC = np.zeros((3*d.mesh.npoints,idx),dtype=np.double)
X_VELOC[0:3*d.mesh.npoints:3,:] = X[1:4*d.mesh.npoints:4,-idx:]
X_VELOC[1:3*d.mesh.npoints:3,:] = X[2:4*d.mesh.npoints:4,-idx:]
X_VELOC[2:3*d.mesh.npoints:3,:] = X[3:4*d.mesh.npoints:4,-idx:]
X_VELOR = np.zeros((3*d.mesh.npoints,idx),dtype=np.double)
X_VELOR[0:3*d.mesh.npoints:3,:] = X_POD[1:4*d.mesh.npoints:4,-idx:]
X_VELOR[1:3*d.mesh.npoints:3,:] = X_POD[2:4*d.mesh.npoints:4,-idx:]
X_VELOR[2:3*d.mesh.npoints:3,:] = X_POD[3:4*d.mesh.npoints:4,-idx:]

d2 = pyLOM.Dataset(ptable=d.partition_table, mesh=d.mesh, time=time)
d2.add_variable('PRESS',True,1,X_PRESS)
d2.add_variable('VELOC',True,3,X_VELOC)
d2.add_variable('PRESR',True,1,X_PRESR)
d2.add_variable('VELOR',True,3,X_VELOR)
d2.write('flow',basedir='flow',instants=np.arange(d2.time.shape[0],dtype=np.int32),times=d2.time,vars=['PRESS','VELOC','PRESR','VELOR'],fmt='vtkh5')


## Show and print timings
pyLOM.cr_info()