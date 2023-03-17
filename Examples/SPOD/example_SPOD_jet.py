#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np
import pyLOM


## Parameters
DATAFILE = 'Examples/Data/jetLES.h5'
VARIABLE = 'PRESS'


## Data loading
d     = pyLOM.Dataset.load(DATAFILE)
X     = d[VARIABLE][:,::10]
t     = d.time[::10]
npwin = 150 #Number of snapshots in each window
nolap = 50 #Number of overlapping snapshots between windows


## Run SPOD
L, P, f = pyLOM.SPOD.run(X,t,nDFT=npwin,nolap=nolap,remove_mean=True)
#pyLOM.SPOD.save('results.h5',L,P,f,d.partition_table,nvars=2,pointData=False)
if pyLOM.utils.is_rank_or_serial(root=0): pyLOM.SPOD.plotSpectra(f, L)


## Dump to ParaView
# Spatial modes
d.add_variable('spatial_modes_U',True,6,pyLOM.SPOD.extract_modes(L,P,1,d.mesh.npoints,modes=[1,2,3,4,5,6]))
d.write('modes',basedir='out/modes',instants=[0],times=[0.],vars=['spatial_modes_U'],fmt='vtkh5')
#pyLOM.SPOD.plotMode(L,P,f,d,1,pointData=True,modes=[1,2,3,4,5,6],cpos='xy')


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()
