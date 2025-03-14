#!/usr/bin/env python
#
# Example of DMD.
#
# Last revision: 27/01/2023
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np
import pyLOM

pyLOM.gpu_device(gpu_per_node=4)


## Parameters
DATAFILE = './DATA/jetLES.h5'
VARIABLE = 'PRESS'


## Data loadingx
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table).to_gpu([VARIABLE])
X = d[VARIABLE][:,::10].copy()
t = d.get_variable('time')[::10].copy()
dt = t[1] - t[0]


## Run DMD
Y = X[:,:100].copy()
muReal,muImag,Phi,bJov = pyLOM.DMD.run(Y,1e-6,remove_mean=False)
# Compute frequency and damping ratio of the modes
delta, omega = pyLOM.DMD.frequency_damping(muReal,muImag,dt)
pyLOM.DMD.save('results.h5',muReal,muImag,Phi,bJov,d.partition_table,nvars=1,pointData=d.point)
# Reconstruction according to Jovanovic 2014
X_DMD = pyLOM.DMD.reconstruction_jovanovic(Phi,muReal,muImag,t,bJov)
rmse  = pyLOM.math.RMSE(X_DMD.copy(),X.copy())
pyLOM.pprint(0,'RMSE = %e'%rmse)


## DMD plots
if pyLOM.utils.is_rank_or_serial(0):
	pyLOM.DMD.ritzSpectrum(muReal,muImag)               # Ritz Spectrum
	pyLOM.DMD.amplitudeFrequency(omega,bJov,norm=False) # Amplitude vs frequency
	pyLOM.DMD.dampingFrequency(omega,delta)             # Damping ratio vs frequency


## Dump to ParaView
# Spatial modes
d.add_field('P_MODES_REAL',6,pyLOM.DMD.extract_modes(Phi,1,len(d),real=True,modes=[1,4,6,2,5,3]))
d.add_field('P_MODES_IMAG',6,pyLOM.DMD.extract_modes(Phi,1,len(d),real=False,modes=[1,4,6,2,5,3]))
pyLOM.io.pv_writer(m,d.to_cpu(['P_MODES_REAL','P_MODES_IMAG']),'modes',basedir='out/modes',instants=[0],times=[0.],vars=['P_MODES_REAL','P_MODES_IMAG'],fmt='vtkh5')
pyLOM.DMD.plotMode(Phi,omega,m,d,1,pointData=True,modes=[1,2,3,4,5,6,7],cpos='xy')

# Temporal evolution
d.add_field('PRESR',1,X_DMD)
pyLOM.io.pv_writer(m,d.to_cpu(['PRESS','PRESR']),'flow',basedir='out/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['PRESS','PRESR'],fmt='vtkh5')
pyLOM.DMD.plotSnapshot(m,d.to_cpu(['PRESR']),vars=['PRESR'],instant=0,cmap='jet',cpos='xy')

# Prediction
t_new = t[-1] + dt*np.arange(0,100,1,np.double)
X_DMD = pyLOM.DMD.reconstruction_jovanovic(Phi,muReal,muImag,t_new,bJov)
d.add_field('PRENR',1,X_DMD)
pyLOM.io.pv_writer(m,d.to_cpu(['PRENR']),'pred',basedir='out/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t_new,vars=['PRENR'],fmt='vtkh5')
pyLOM.DMD.plotSnapshot(m,d.to_cpu(['PRENR']),vars=['PRENR'],instant=40,cmap='jet',cpos='xy')


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()
