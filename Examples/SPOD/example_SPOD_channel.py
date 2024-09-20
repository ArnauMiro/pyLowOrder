#!/usr/bin/env python
#
# SPOD analysis.
#
# Last revision: 29/10/2021
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np
import pyLOM


## Data loading
DATAFILE  = './channel.h5'
VARIABLES = ['PRESS','VELOX','VELOY','VELOZ']

m     = pyLOM.Mesh.load(DATAFILE)
d     = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X     = d.X(*VARIABLES)
npwin = 100 # Number of snapshots in each window
nolap = 20  # Number of overlapping snapshots between windows


## Run SPOD
L, P, f = pyLOM.SPOD.run(X,t,nDFT=npwin,nolap=nolap,remove_mean=True)
pyLOM.SPOD.save('results.h5',L,P,f,d.partition_table,nvars=4,pointData=d.point)
if pyLOM.utils.is_rank_or_serial(root=0): 
	fig,_ = pyLOM.SPOD.plotSpectra(f, L)
	fig.savefig('spectra.png',dpi=300)


## Dump to ParaView
# Spatial modes
modes = np.arange(1,10+1,dtype=np.int32)
d.add_field('spatial_modes_P',len(modes),pyLOM.SPOD.extract_modes(L,P,1,len(d),modes=modes))
d.add_field('spatial_modes_U',len(modes),pyLOM.SPOD.extract_modes(L,P,2,len(d),modes=modes))
d.add_field('spatial_modes_V',len(modes),pyLOM.SPOD.extract_modes(L,P,3,len(d),modes=modes))
d.add_field('spatial_modes_W',len(modes),pyLOM.SPOD.extract_modes(L,P,4,len(d),modes=modes))
pyLOM.io.pv_writer(m,d,'modes',basedir='modes',instants=[0],times=[0.],vars=['spatial_modes_P','spatial_modes_U','spatial_modes_V','spatial_modes_W'],fmt='vtkh5')


## Show and print timings
pyLOM.cr_info()