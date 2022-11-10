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
remove_mean = False
r = 20
pyLOM.cr_start('example',0)
t = np.arange(0, 151, dtype = np.double)

#Run DMD routine
Y = X[:,:100].copy()
muReal, muImag, Phi, bJov = pyLOM.DMD.run(Y, r, remove_mean)

#Compute frequency and damping ratio of the modes
delta, omega = pyLOM.DMD.frequency_damping(muReal, muImag, dt)

#Reconstruction according to Jovanovic 2014
X_DMD = pyLOM.DMD.reconstruction_jovanovic(Phi, muReal, muImag, t, bJov)
rmse = pyLOM.math.RMSE(X_DMD.copy(), X.copy())
print('RMSE = %e' % rmse)

#Ritz Spectrum
pyLOM.DMD.ritzSpectrum(muReal, muImag)

#Amplitude vs frequency
pyLOM.DMD.amplitudeFrequency(omega, bJov, norm = False)

#Damping ratio vs frequency
pyLOM.DMD.dampingFrequency(omega, delta)

#Plot modes and reconstructed flow
pyLOM.DMD.plotMode(Phi, muImag/dt, d.xyz, d.mesh, d.info(VARIABLE), modes = [1, 2, 3, 4, 5, 6, 7]) #MATLAB: modes 17, 112 and 144
fig,ax,anim = pyLOM.DMD.animateFlow(X,X_DMD,d.xyz,d.mesh,d.info(VARIABLE),dim=0)

## Show and print timings
pyLOM.cr_stop('example',0)
pyLOM.cr_info()
#pyLOM.show_plots()
