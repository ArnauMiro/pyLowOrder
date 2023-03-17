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
d     = pyLOM.Dataset.load(DATAFILE)
X     = d[VARIABLE]
t     = d.time
npwin = 50 #Number of snapshots in each window
nolap = 10 #Number of overlapping snapshots between windows


## Run SPOD
L, P, f = pyLOM.SPOD.run(X,t,nDFT=npwin,nolap=nolap,remove_mean=True)
if pyLOM.utils.is_rank_or_serial(root=0):
    fig,_ = pyLOM.SPOD.plotSpectra(f, L)
    os.makedirs('jetSPOD',exist_ok=True)
    fig.savefig('jetSPOD/spectra.png',dpi=300)
pyLOM.SPOD.save('jetSPOD/results.h5',L,P,f,d.partition_table,nvars=1,pointData=True)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_U',True,6,pyLOM.SPOD.extract_modes(L,P,1,d.mesh.npoints,modes=[1,2,3,4,5,6]))
d.write('modes',basedir='jetSPOD/modes',instants=[0],times=[0.],vars=['spatial_modes_U'],fmt='vtkh5')


## Show and print timings
pyLOM.cr_info()