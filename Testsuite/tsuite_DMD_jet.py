#!/usr/bin/env python
#
# PYLOM Testsuite
# Run DMD on the jet dataset
#
# Last revision: 27/01/2023
from __future__ import print_function, division

import os, numpy as np
import pyLOM


## Parameters
DATAFILE = './JET.h5'
VARIABLE = 'PRESS'


## Data loadingx
d = pyLOM.Dataset.load(DATAFILE)
X  = d[VARIABLE]
t  = d.time
dt = t[1] - t[0]


## Run DMD
Y = X[:,:100].copy() # Grab the first 100 and run DMD
muReal,muImag,Phi,bJov = pyLOM.DMD.run(Y,1e-6,remove_mean=False)
# Compute frequency and damping ratio of the modes
delta, omega = pyLOM.DMD.frequency_damping(muReal,muImag,dt)
os.makedirs('jetDMD',exist_ok=True)
pyLOM.DMD.save('jetDMD/results.h5',muReal,muImag,Phi,bJov,d.partition_table,nvars=1,pointData=True)
# Reconstruction according to Jovanovic 2014
# Last 50 snapshots are predicted
X_DMD = pyLOM.DMD.reconstruction_jovanovic(Phi,muReal,muImag,t,bJov)
rmse  = pyLOM.math.RMSE(X_DMD.copy(),X.copy())
pyLOM.pprint(0,'RMSE = %e'%rmse)


## DMD plots
if pyLOM.utils.is_rank_or_serial(0):
	fig,_ = pyLOM.DMD.ritzSpectrum(muReal,muImag)               # Ritz Spectrum
	fig.savefig('jetDMD/ritzSpectrum.png',dpi=300)
	fig,_ = pyLOM.DMD.amplitudeFrequency(omega,bJov,norm=False) # Amplitude vs frequency
	fig.savefig('jetDMD/amplitudeFrequency.png',dpi=300)
	fig,_ = pyLOM.DMD.dampingFrequency(omega,delta)             # Damping ratio vs frequency
	fig.savefig('jetDMD/dampingFrequency.png',dpi=300)


## Dump to ParaView
# Spatial modes
d.add_variable('P_MODES_REAL',True,6,0,pyLOM.DMD.extract_modes(Phi,1,d.mesh.npoints,real=True,modes=[1,4,6,2,5,3]))
d.add_variable('P_MODES_IMAG',True,6,0,pyLOM.DMD.extract_modes(Phi,1,d.mesh.npoints,real=False,modes=[1,4,6,2,5,3]))
d.write('modes',basedir='jetDMD/modes',instants=[0],times=[0.],vars=['P_MODES_REAL','P_MODES_IMAG'],fmt='vtkh5')
#pyLOM.DMD.plotMode(Phi,omega,d,1,pointData=True,modes=[1,2,3,4,5,6,7],cpos='xy',off_screen=True,screenshot='jetDMD/mode_P_%d'%(pyLOM.utils.MPI_RANK)+'_%d.png')

# Temporal evolution
d.add_variable('PRESR',True,1,t.shape[0],X_DMD)
d.write('flow',basedir='jetDMD/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['PRESS','PRESR'],fmt='vtkh5')
#pyLOM.DMD.plotSnapshot(d,vars=['PRESR'],instant=0,cmap='jet',cpos='xy',off_screen=True,screenshot='jetDMD/P_%d.png'%pyLOM.utils.MPI_RANK)

# Prediction
t_new = t[-1] + dt*np.arange(0,100,1,np.double)
X_DMD = pyLOM.DMD.reconstruction_jovanovic(Phi,muReal,muImag,t_new,bJov)
d.add_variable('PRENR',True,1,t_new.shape[0],X_DMD)
d.write('pred',basedir='jetDMD/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t_new,vars=['PRENR'],fmt='vtkh5')
#pyLOM.DMD.plotSnapshot(d,vars=['PRENR'],instant=0,cmap='jet',cpos='xy',off_screen=True,screenshot='jetDMD/P_new_%d.png'%pyLOM.utils.MPI_RANK)


## Show and print timings
pyLOM.cr_info()