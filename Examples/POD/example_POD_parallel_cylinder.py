#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os
import numpy as np
import pyLOM
from pyLOM.utils.parall import mpi_gather


## Parameters
DATAFILE = './Examples/Data/CYLINDER.h5'
VARIABLE = 'VELOC'

## Data loading
d = pyLOM.Dataset.load(DATAFILE)
X  = d[VARIABLE]
t  = d.time

## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False) # PSI are POD modes
if pyLOM.is_rank_or_serial(root=0): pyLOM.POD.plotResidual(S)
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=5e-6)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_U',False,6,0,pyLOM.POD.extract_modes(PSI,1,d.ncells,modes=[1,4,6,2,5,3]))
d.add_variable('spatial_modes_V',False,6,0,pyLOM.POD.extract_modes(PSI,2,d.ncells,modes=[1,4,6,2,5,3]))
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


## Gather to rank 0 for plotting reasons
PSI   = mpi_gather(PSI,root=0)
X     = mpi_gather(X,root=0)
X_POD = mpi_gather(X_POD,root=0)
xyz   = mpi_gather(d.xyz,root=0)

# Plot POD mode
if pyLOM.is_rank_or_serial(0):
	# 0 - module, 1,2 - components
	pyLOM.POD.plotMode(PSI,xyz,V[:,:-1],t[:-1],d.mesh,d.info(VARIABLE),dim=0,modes=[1,2,3,4])

	# Plot reconstructed flow
	fig,ax,anim = pyLOM.POD.animateFlow(X,X_POD,xyz,d.mesh,d.info(VARIABLE),dim=0)


## Show and print timings
pyLOM.cr_info()
if pyLOM.is_rank_or_serial(0): pyLOM.show_plots()
