#!/usr/bin/env python
#
# POD analysis.
#
# Last revision: 29/10/2021
from __future__ import print_function, division

import os, numpy as np
import pyAlya, pyLOM

## Data loading
DATAFILE  = './chan.h5'
VARIABLES = ['PRESS','VELOC']

t_slice = np.s_[:]
d = pyLOM.Dataset.load(DATAFILE)
X = d.X(*VARIABLES,time_slice=t_slice)
t  = d.time[t_slice]


## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
if pyLOM.utils.is_rank_or_serial(root=0): 
	fig,_ = pyLOM.POD.plotResidual(S)
	fig.savefig('residuals.png')
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=3e-2)
pyLOM.POD.save(DATAFILE,PSI,S,V,d.partition_table,nvars=4,mode='a',pointData=True)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)


## Store into pyAlya field
f = pyAlya.Field(xyz = d.mesh.xyz, ptable = d.partition_table,
		P_MODES = pyLOM.POD.extract_modes(PSI,1,d.mesh.npoints,reshape=True),
		U_MODES = pyLOM.POD.extract_modes(PSI,2,d.mesh.npoints,reshape=True),
		V_MODES = pyLOM.POD.extract_modes(PSI,3,d.mesh.npoints,reshape=True),
		W_MODES = pyLOM.POD.extract_modes(PSI,4,d.mesh.npoints,reshape=True),
)
#f.save('POD_modes.h5')


## Dump to ParaView - Spatial modes
for v in f.varnames:
	d.add_variable(v,True,f[v].shape[1],0,f[v])
d.write('modes',basedir='out/modes',instants=[0],times=[0.],vars=['P_MODES','U_MODES','V_MODES','W_MODES'],fmt='vtkh5')

# Temporal evolution
d.add_variable('X_POD',True,4,t.shape[0],X_POD)
d.write('flow',basedir='out/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['VELOC','PRESS','X_POD'],fmt='vtkh5')


## Plot POD mode
if pyLOM.utils.is_rank_or_serial(0):
	modes = np.arange(PSI.shape[1],dtype=np.int32) + 1
	fig,_ = pyLOM.POD.plotMode(V,t,modes=modes)
	for imode,mode in enumerate(modes): fig[imode].savefig('out/mode_%d.png'%mode)


## Show and print timings
pyLOM.cr_info()