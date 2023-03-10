#!/usr/bin/env python
#
# Example of DMD.
#
# Last revision: 27/01/2023
from __future__ import print_function, division

import os, numpy as np
import pyLOM


## Parameters
DATAFILE = 'Examples/Data/CYLINDER.h5'
VARIABLE = 'VELOX'


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
X  = d[VARIABLE]
t  = d.time
dt = t[1] - t[0]


## Run DMD
Y = X[:,:100].copy() # Grab the first 100 and run DMD
muReal,muImag,Phi,bJov = pyLOM.DMD.run(Y,1e-6,remove_mean=False)
# Compute frequency and damping ratio of the modes
delta, omega = pyLOM.DMD.frequency_damping(muReal,muImag,dt)
pyLOM.DMD.save('results.h5',muReal,muImag,Phi,bJov,d.partition_table,nvars=1,pointData=False)
# Reconstruction according to Jovanovic 2014
# Last 50 snapshots are predicted
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
d.add_variable('U_MODES_REAL',False,6,pyLOM.DMD.extract_modes(Phi,1,d.mesh.ncells,real=True,modes=[1,4,6,2,5,3]))
d.add_variable('U_MODES_IMAG',False,6,pyLOM.DMD.extract_modes(Phi,1,d.mesh.ncells,real=False,modes=[1,4,6,2,5,3]))
d.write('modes',basedir='out/modes',instants=[0],times=[0.],vars=['U_MODES_REAL','U_MODES_IMAG'],fmt='vtkh5')
pyLOM.DMD.plotMode(Phi,omega,d,1,pointData=False,modes=[1,2,3,4,5,6,7],cpos='xy')

# Temporal evolution
d.add_variable('VELXR',False,1,X_DMD)
d.write('flow',basedir='out/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['VELOX','VELXR'],fmt='vtkh5')
pyLOM.DMD.plotSnapshot(d,vars=['VELXR'],instant=0,cmap='jet',cpos='xy')


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()