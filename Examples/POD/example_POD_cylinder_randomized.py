#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np
import pyLOM

pyLOM.gpu_device(gpu_per_node=4) # Detect GPU configuration


## Parameters
DATAFILE = './DATA/CYLINDER.h5'
VARIABLE = 'VELOC'


## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table).to_gpu([VARIABLE]) # Send to GPU if available
X = d[VARIABLE]
t = d.get_variable('time')


## Run POD
PSI,S,V = pyLOM.POD.run(X,remove_mean=False,randomized=True,r=8,q=3) # PSI are POD modes
if pyLOM.utils.is_rank_or_serial(root=0): pyLOM.POD.plotResidual(S)
# Truncate to a number of modes
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=6)
pyLOM.POD.save('results.h5',PSI,S,V,d.partition_table,nvars=2,pointData=d.point)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)
pyLOM.pprint(0,'RMSE = %e'%rmse)


## Dump to ParaView
# Spatial modes
d.add_field('spatial_modes_U',6,pyLOM.POD.extract_modes(PSI,1,len(d),modes=[1,4,6,2,5,3]))
d.add_field('spatial_modes_V',6,pyLOM.POD.extract_modes(PSI,2,len(d),modes=[1,4,6,2,5,3]))
pyLOM.io.pv_writer(m,d.to_cpu(['spatial_modes_U','spatial_modes_V']),'modes',basedir='out/modes',instants=[0],times=[0.],vars=['spatial_modes_U','spatial_modes_V'],fmt='vtkh5')
pyLOM.POD.plotSnapshot(m,d.to_cpu(['spatial_modes_U']),vars=['spatial_modes_U'],instant=0,component=0,cmap='jet',cpos='xy')

# Temporal evolution
d.add_field('VELOR',2,X_POD)
pyLOM.io.pv_writer(m,d.to_cpu(['VELOC','VELOR']),'flow',basedir='out/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['VELOC','VELOR'],fmt='vtkh5')
pyLOM.POD.plotSnapshot(m,d.to_cpu(['VELOR']),vars=['VELOR'],instant=0,component=0,cmap='jet',cpos='xy')


## Plot POD mode
if pyLOM.utils.is_rank_or_serial(0):
	# 0 - module, 1,2 - components
	pyLOM.POD.plotMode(V[:,:-1],t[:-1],modes=[1,2,3,4])


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()