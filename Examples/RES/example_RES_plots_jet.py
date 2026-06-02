from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import os, numpy as np
import matplotlib.pyplot as plt
import pyLOM

pyLOM.gpu_device(gpu_per_node=4)


# Parameters
DATAFILE = './DATA/jetLES.h5'
VARIABLE = 'PRESS'

param = 2 * np.pi # Define the normalization parameter for the frequency (in St)
f_list = [0.2, 0.4, 0.6, 0.8, 1.0] # Desired frequencies (in St)
n_modes = 5 # Desired number of modes to save 
modes = np.arange(1,n_modes+1,dtype=np.int32)


# Load the mesh
m = pyLOM.Mesh.load(DATAFILE)
pyLOM.pprint(0,'mesh loaded', flush=True)


# Load the dataset
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table).to_gpu([VARIABLE])
X = d[VARIABLE]
t = d.get_variable('time')
dt = t[1] - t[0]
pyLOM.pprint(0,'dataset loaded', flush=True)


# Compute the DMD of the case
muReal, muImag, Phi, bJov = pyLOM.DMD.run(X, r=2*10**-1, remove_mean=True)
delta, omega = pyLOM.DMD.frequency_damping(muReal,muImag,dt)
freq = omega / param


S_list = np.empty((0,Phi.shape[1]))
# Compute the Resolvent Analysis for every frequency
for f in f_list:
    U, S, V = pyLOM.RES.run(Phi, delta, freq, f=f, Q=None)
    S_list = np.vstack([S_list, S])

    # Extract the desired modes
    U2, V2 = pyLOM.RES.extract_modes(U,V,1,len(d),modes=modes,kind='real') # kind can be 'real', 'imag' or 'abs'


    # Write the modes to be visualized with paraview
    d.add_field(f'forcing_modes',len(modes),V2)
    d.add_field(f'response_modes',len(modes),U2)
    pyLOM.io.pv_writer(m,d.to_cpu(['forcing_modes','response_modes']),f'modes_{f}',basedir='./',instants=[0],times=[0.],vars=['forcing_modes','response_modes'],fmt='vtkh5')


if pyLOM.utils.is_rank_or_serial(0):
    # Plot the cumulative energy gains
    pyLOM.RES.plotEnergy(S_list[0,:])
    plt.savefig('energy.png', dpi=300)

    # Plot the energy gains vs frequency for the desired modes
    pyLOM.RES.plotEvW(S_list, f_list, modes)
    plt.savefig('EvW.png', dpi=300)



pyLOM.cr_info()
pyLOM.show_plots()
