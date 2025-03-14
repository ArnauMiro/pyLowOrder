#!/usr/bin/env python
#
# Example of SPOD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import pyLOM

pyLOM.gpu_device(gpu_per_node=4)


## Parameters
DATAFILE = './DATA/jetLES.h5'
VARIABLE = 'PRESS'


## Data loading
m     = pyLOM.Mesh.load(DATAFILE)
d     = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table).to_gpu([VARIABLE])
X     = d[VARIABLE][:,::10].copy()
t     = d.get_variable('time')[::10].copy()
npwin = 150 #Number of snapshots in each window
nolap = 50 #Number of overlapping snapshots between windows


## Run SPOD
L, P, f = pyLOM.SPOD.run(X,t,nDFT=npwin,nolap=nolap,remove_mean=True)
pyLOM.SPOD.save('results.h5',L,P,f,d.partition_table,nvars=1,pointData=d.point)
if pyLOM.utils.is_rank_or_serial(root=0): pyLOM.SPOD.plotSpectra(f, L)


## Dump to ParaView
# Spatial modes
d.add_field('spatial_modes_U',6,pyLOM.SPOD.extract_modes(L,P,1,len(d),modes=[1,2,3,4,5,6]))
pyLOM.io.pv_writer(m,d.to_cpu(['spatial_modes_U']),'modes',basedir='out/modes',instants=[0],times=[0.],vars=['spatial_modes_U'],fmt='vtkh5')
pyLOM.SPOD.plotMode(L,P,f,m,d,1,pointData=d.point,modes=[1,2,3,4,5,6],cpos='xy')


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()