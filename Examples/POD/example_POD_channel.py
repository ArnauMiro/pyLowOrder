#!/usr/bin/env python
#
# POD analysis.
#
# Last revision: 29/10/2021
from __future__ import print_function, division

import os, numpy as np
import pyLOM


## Data loading
DATAFILE  = '../channel.h5'
VARIABLES = ['PRESS','VELOC']

d = pyLOM.Dataset.load(DATAFILE)
X = d.X(*VARIABLES)


## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
if pyLOM.utils.is_rank_or_serial(root=0): 
	fig,_ = pyLOM.POD.plotResidual(S)
	fig.savefig('residuals.png',dpi=300)

# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=5e-6)

# Store results
pyLOM.POD.save('results.h5',PSI,S,V,d.partition_table,nvars=4,pointData=True)

# Spatial modes
d.add_variable('spatial_modes_P',True,PSI.shape[1],0,pyLOM.POD.extract_modes(PSI,1,d.mesh.npoints))
d.add_variable('spatial_modes_U',True,PSI.shape[1],0,pyLOM.POD.extract_modes(PSI,2,d.mesh.npoints))
d.add_variable('spatial_modes_V',True,PSI.shape[1],0,pyLOM.POD.extract_modes(PSI,3,d.mesh.npoints))
d.add_variable('spatial_modes_W',True,PSI.shape[1],0,pyLOM.POD.extract_modes(PSI,4,d.mesh.npoints))
d.write('modes',basedir='modes',instants=[0],times=[0.],vars=['spatial_modes_P','spatial_modes_U','spatial_modes_V','spatial_modes_W'],fmt='vtkh5')

# Plot POD modes
if pyLOM.utils.is_rank_or_serial(0):
	# 0 - module, 1,2 - components
	os.makedirs('modes',exist_ok=True)
	figs,_ = pyLOM.POD.plotMode(V[:,:-1],t[:-1],modes=[1,2,3,4])
	for ifig,fig in enumerate(figs): fig.savefig('modes/mode_%d.png'%ifig) 


## Reconstruct the flow
time  = d.time[-500:]
X_POD = pyLOM.POD.reconstruct(PSI[:,-500:],S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X[:,-500:])
pyLOM.pprint(0,'RMSE = %e'%rmse)

# Create a dataset to store the last 500 instants from the simulation
# and the reconstructed flow
d2   = pyLOM.Dataset(ptable=d.partition_table, mesh=d.mesh, time=time)
d2.add_variable('PRESS',True,1,500,d.X('PRESS',time_slice=np.s_[-500:]))
d2.add_variable('VELOC',True,3,500,d.X('VELOC',time_slice=np.s_[-500:]))
d2.add_variable('PRESR',True,1,500,X_POD[0:4*d.mesh.npoints:4,:])
d2.add_variable('VELXR',True,1,500,X_POD[1:4*d.mesh.npoints:4,:])
d2.add_variable('VELYR',True,1,500,X_POD[2:4*d.mesh.npoints:4,:])
d2.add_variable('VELZR',True,1,500,X_POD[3:4*d.mesh.npoints:4,:])
d2.write('flow',basedir='flow',instants=np.arange(d2.t.shape[0],dtype=np.int32),times=d2.time,vars=['PRESS','VELOC','PRESR','VELXR','VELYR','VELZR'],fmt='vtkh5')


## Show and print timings
pyLOM.cr_info()