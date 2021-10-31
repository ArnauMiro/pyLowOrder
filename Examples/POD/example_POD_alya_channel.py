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

d = pyLOM.Dataset.load(DATAFILE)
X = d.X(*VARIABLES)

## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False)
if pyLOM.is_rank_or_serial(0): 
	fig,_   = pyLOM.POD.plotResidual(S)
	fig.savefig('residuals.png')
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=3e-2)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)

# Generate a pyAlya Field class with the output modes
f = pyAlya.Field(xyz = d.xyz,
		LNINV   = d.pointOrder, # We need this so that we can repartition later
		P_MODES = pyLOM.POD.extract_modes(PSI,1,d.npoints,reshape=True),
		U_MODES = pyLOM.POD.extract_modes(PSI,2,d.npoints,reshape=True),
		V_MODES = pyLOM.POD.extract_modes(PSI,3,d.npoints,reshape=True),
		W_MODES = pyLOM.POD.extract_modes(PSI,4,d.npoints,reshape=True),
)
f.save('POD_modes.h5',write_master=True) # Only Alya has a master


## Plots
if pyLOM.is_rank_or_serial(0): 
	modes   = np.arange(PSI.shape[1],dtype=np.int32) + 1
	fig,_,_ = pyLOM.POD.plotMode(PSI,d.xyz,V,d.time,d.mesh,d.info('VELOC'),dim=0,modes=modes)
	for imode,mode in enumerate(modes): fig[imode].savefig('mode_%d.png'%mode)


## Show and print timings
pyLOM.cr_info()
