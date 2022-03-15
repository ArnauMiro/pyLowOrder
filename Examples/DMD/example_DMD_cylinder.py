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
import sys

## Data loading
d  = pyLOM.Dataset.load('Examples/Data/CYLINDER.h5')
VARIABLE = 'VELOX'
X  = d[VARIABLE]
dt = 0.2
remove_mean = True
r = 1e-6
validation = scipy.io.loadmat('/media/sf_TUAREG/TOOLS/DMD_algorithms_BSC/DMD/validation.mat')
pyLOM.cr_start('example',0)
#Run DMD routine
muReal, muImag, w, Phi = pyLOM.DMD.run(X, r, remove_mean)

#Compute frequency and damping ratio of the modes
#delta, omega = pyLOM.DMD.frequency_damping(muReal, muImag, dt)

#Reconstruction according to Jovanovic 2014
'''
Y_DMD = pyLOM.DMD.reconstruction_jovanovic(PSI, w, muReal, muImag, Y[:, :-1], bJov)
rmse  = pyLOM.math.RMSE(Y_DMD.real, Y[:, :-1])
print('RMSE = %e' % rmse)
'''

#Ritz Spectrum
pyLOM.DMD.ritzSpectrum(muReal, muImag)

#Amplitude vs frequency
#pyLOM.DMD.amplitudeFrequency(omega, bJov, norm = False)

#Damping ratio vs frequency
#pyLOM.DMD.dampingFrequency(omega, delta)

#Plot modes and reconstructed flow
pyLOM.DMD.plotMode(Phi, muImag/dt, d.xyz, d.mesh, d.info(VARIABLE), modes = [1, 3, 17, 112, 114]) #MATLAB: modes 17, 112 and 144
#fig,ax,anim = pyLOM.DMD.animateFlow(Y,Y_DMD.real,d.xyz,d.mesh,d.info(VARIABLE),dim=0)

## Show and print timings
pyLOM.cr_stop('example',0)
pyLOM.cr_info()
pyLOM.show_plots()
