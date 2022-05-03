#!/usr/bin/env python
#
# Example of POD following the MATLAB script.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io
import pyLOM
from pyLOM.utils.parall import mpi_gather
import sys

## Data loading
d  = pyLOM.Dataset.load('Examples/Data/CYLINDER.h5')
VARIABLE = 'VELOX'
X  = d[VARIABLE]
dt = 0.2
remove_mean = False
r = 5e-6
validation = scipy.io.loadmat('../DMD_algorithms_BSC/DMD/amplitudeValidation.mat')
pyLOM.cr_start('example',0)

#Run DMD routine
Ur, muReal, muImag, w, Phi, bJov = pyLOM.DMD.run(X, r, remove_mean)

#Reconstruction according to Jovanovic 2014
X_DMD = pyLOM.DMD.reconstruction_jovanovic(Ur, w, muReal, muImag, X, bJov)

#Gather results
phi    = mpi_gather(Phi,root=0)
xyz    = mpi_gather(d.xyz,root=0)
X_DMDG = mpi_gather(X_DMD.real,root=0)
XG     = mpi_gather(X,root=0)

#Compute frequency and damping ratio of the modes
if pyLOM.is_rank_or_serial(0):
    delta, omega = pyLOM.DMD.frequency_damping(muReal, muImag, dt)

    #Ritz Spectrum
    pyLOM.DMD.ritzSpectrum(muReal, muImag)

    #Amplitude vs frequency
    pyLOM.DMD.amplitudeFrequency(omega, bJov, norm = False)

    #Damping ratio vs frequency
    pyLOM.DMD.dampingFrequency(omega, delta)

    #Plot modes and reconstructed flow
    pyLOM.DMD.plotMode(phi, muImag/dt, xyz, d.mesh, d.info(VARIABLE), modes = [0, 1, 2, 3, 4, 5]) #MATLAB: modes 17, 112 and 144
    fig,ax,anim = pyLOM.DMD.animateFlow(XG[:, :-1],X_DMDG,xyz,d.mesh,d.info(VARIABLE),dim=0)

    ## Show and print timings
    pyLOM.show_plots()
pyLOM.cr_stop('example',0)
pyLOM.cr_info()
