#!/usr/bin/env python
#
# DMD analysis.
#
# Last revision: 29/10/2021
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import os, numpy as np
import pyLOM


## Data loading
DATAFILE  = '../channel.h5'
VARIABLES = ['VELOC']

d  = pyLOM.Dataset.load(DATAFILE)
X  = d.X(*VARIABLES)[0:3*d.mesh.npoints:3,:] # Select only X component
dt = d.time[1] - d.time[0]


## Run DMD
muReal,muImag,Phi,bJov = pyLOM.DMD.run(X,1e-6,remove_mean=False)
# Compute frequency and damping ratio of the modes
delta, omega = pyLOM.DMD.frequency_damping(muReal,muImag,dt)
pyLOM.DMD.save('results.h5',muReal,muImag,Phi,bJov,d.partition_table,nvars=1,pointData=True)

# plots
if pyLOM.utils.is_rank_or_serial(0):
	fig,_ = pyLOM.DMD.ritzSpectrum(muReal,muImag)               # Ritz Spectrum
	fig.savefig('ritzSpectrum.png',dpi=300)
	fig,_ = pyLOM.DMD.amplitudeFrequency(omega,bJov,norm=False) # Amplitude vs frequency
	fig.savefig('amplitudeFrequency.png',dpi=300)
	fig,_ = pyLOM.DMD.dampingFrequency(omega,delta)             # Damping ratio vs frequency
	fig.savefig('dampingFrequency.png',dpi=300)

# Spatial modes
modes = np.arange(1,100+1,dtype=np.int32)
d.add_variable('U_MODES_REAL',True,len(modes),pyLOM.DMD.extract_modes(Phi,1,d.mesh.npoints,modes=modes,real=True))
d.add_variable('U_MODES_IMAG',True,len(modes),pyLOM.DMD.extract_modes(Phi,1,d.mesh.npoints,modes=modes,real=False))
d.write('modes',basedir='modes',instants=[0],times=[0.],vars=['U_MODES_REAL','U_MODES_IMAG'],fmt='vtkh5')


## Prediction
t_new = d.time[-1] + dt*np.arange(0,100,1,np.double)
X_DMD = pyLOM.DMD.reconstruction_jovanovic(Phi,muReal,muImag,t_new,bJov)
d.add_variable('VELXP',True,1,X_DMD)
d.write('pred',basedir='flow',instants=np.arange(t_new.shape[0],dtype=np.int32),times=t_new,vars=['VELXP'],fmt='vtkh5')


## Show and print timings
pyLOM.cr_info()