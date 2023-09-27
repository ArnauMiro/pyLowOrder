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

d     = pyLOM.Dataset.load(DATAFILE)
X     = d.X(*VARIABLES)
t     = d.time
npwin = 100 # Number of snapshots in each window
nolap = 20  # Number of overlapping snapshots between windows


## Run SPOD
L, P, f = pyLOM.SPOD.run(X,t,nDFT=npwin,nolap=nolap,remove_mean=True)
pyLOM.SPOD.save('results.h5',L,P,f,d.partition_table,nvars=4,pointData=True)
if pyLOM.utils.is_rank_or_serial(root=0): 
	fig,_ = pyLOM.SPOD.plotSpectra(f, L)
	fig.savefig('spectra.png',dpi=300)


## Dump to ParaView
# Spatial modes
modes = np.arange(1,10+1,dtype=np.int32)
d.add_variable('spatial_modes_P',True,len(modes),pyLOM.SPOD.extract_modes(L,P,1,d.mesh.npoints,modes=modes))
d.add_variable('spatial_modes_U',True,len(modes),pyLOM.SPOD.extract_modes(L,P,2,d.mesh.npoints,modes=modes))
d.add_variable('spatial_modes_V',True,len(modes),pyLOM.SPOD.extract_modes(L,P,3,d.mesh.npoints,modes=modes))
d.add_variable('spatial_modes_W',True,len(modes),pyLOM.SPOD.extract_modes(L,P,4,d.mesh.npoints,modes=modes))
d.write('modes',basedir='modes',instants=[0],times=[0.],vars=['spatial_modes_P','spatial_modes_U','spatial_modes_V','spatial_modes_W'],fmt='vtkh5')


## Show and print timings
pyLOM.cr_info()
