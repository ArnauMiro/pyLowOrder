from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import os, numpy as np
import pyLOM


# Parameters
DATAFILE = './DATA/jetLES.h5'
VARIABLES = 'PRESS'

param = 2 * np.pi # Define the normalization parameter for the frequency (in St)
f = 0.8 # Desired frequency (in St)
n_modes = 5 # Desired number of modes to save 
modes = np.arange(1,n_modes+1,dtype=np.int32)


# Load the mesh
m = pyLOM.Mesh.load(DATAFILE)
pyLOM.pprint(0,'mesh loaded', flush=True)


# Load the dataset
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d[VARIABLES]
t = d.get_variable('time')
dt = t[1] - t[0]
pyLOM.pprint(0,'dataset loaded', flush=True)


# Compute the DMD of the case
muReal, muImag, Phi, bJov = pyLOM.DMD.run(X, r=2e-1, remove_mean=True)
delta, omega = pyLOM.DMD.frequency_damping(muReal,muImag,dt)
freq = omega / param


# Compute the Resolvent Analysis
U, S, V = pyLOM.RES.run(Phi, delta, freq, f=f, Q=None)


# Extract the desired modes
U2, V2 = pyLOM.RES.extract_modes(U,V,1,len(d),modes=modes,kind='real') # kind can be 'real', 'imag' or 'abs'


# Write the modes to be visualized with paraview
d.add_field(f'forcing_modes',len(modes),V2)
d.add_field(f'response_modes',len(modes),U2)
pyLOM.io.pv_writer(m,d,f'modes_{f}',basedir='./',instants=[0],times=[0.],vars=[f'forcing_modes',f'response_modes'],fmt='vtkh5')


pyLOM.cr_info()
