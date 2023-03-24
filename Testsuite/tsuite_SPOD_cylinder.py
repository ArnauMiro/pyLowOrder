#!/usr/bin/env python
#
# PYLOM Testsuite
# Run SPOD on the cylinder dataset
#
# Last revision: 17/03/2023
from __future__ import print_function, division

import os, numpy as np
import pyLOM


## Parameters
DATAFILE = './CYLINDER.h5'
VARIABLE = 'VELOX'


## Data loading
d     = pyLOM.Dataset.load(DATAFILE)
X     = d[VARIABLE]
t     = d.time
npwin = 8 #Number of snapshots in each window
nolap = 2 #Number of overlapping snapshots between windows


## Run SPOD
L, P, f = pyLOM.SPOD.run(X,t,nDFT=npwin,nolap=nolap,remove_mean=True)
if pyLOM.utils.is_rank_or_serial(root=0): 
    fig,_ = pyLOM.SPOD.plotSpectra(f, L)
    os.makedirs('cylinderSPOD',exist_ok=True)
    fig.savefig('cylinderSPOD/spectra.png',dpi=300)
pyLOM.SPOD.save('cylinderSPOD/results.h5',L,P,f,d.partition_table,nvars=2,pointData=False)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_U',False,6,pyLOM.SPOD.extract_modes(L,P,1,d.mesh.ncells,modes=[1,2,3,4,5,6]))
d.write('modes',basedir='cylinderSPOD/modes',instants=[0],times=[0.],vars=['spatial_modes_U'],fmt='vtkh5')


## Show and print timings
pyLOM.cr_info()
