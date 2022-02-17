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

## Data loading
d  = pyLOM.Dataset.load('Examples/Data/CYLINDER.h5')
X  = d['VELOX']
dt = 0.2
#scipy.io.savemat('data.m', {'X' : X})
validation = scipy.io.loadmat('/media/sf_TUAREG/TOOLS/DMD_algorithms_BSC/DMD/validation2.mat')

#fig, ax = plt.subplots(3,1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k',gridspec_kw = {'hspace':0.5})
#pyLOM.cr_start('example',0)

#Compute and substract temporal subtract_mean
Xavg = pyLOM.math.temporal_mean(X)
X  = pyLOM.math.subtract_mean(X, Xavg)
PSI, S, V = pyLOM.math.tsqr_svd(X[:, :-1])

# Truncate according to residual
PSI, S, V = pyLOM.POD.truncate(PSI, S, V, r = 1e-6)

#Project A (Jacobian of the snapshots) into POD basis
aux1 = pyLOM.math.matmul(pyLOM.math.transpose(PSI),X[:,1:])
aux2 = pyLOM.math.transpose(pyLOM.math.vecmat(1./S,V))
Atilde = pyLOM.math.matmul(aux1,aux2)

#Eigendecomposition of Atilde
muReal, muImag, w = pyLOM.math.eigen(Atilde)

#Compute frequency and damping ratio of the modes
delta, omega = pyLOM.DMD.frequency_damping(muReal, muImag, dt)

#Computation of the modes
Phi = pyLOM.DMD.mode_computation(X[:, 1:], V, S, w)

#Computation of the amplitudes according to Jovanovic 2014
bJov = pyLOM.DMD.amplitude_jovanovic(muReal, muImag, X[:, :-1], w, S, V)

#Reconstruction according to Jovanovic 2014
#Xdmd = pyLOM.DMD.reconstruction_jovanovic(U, w, muReal, muImag, X[:, :-1], bJov)

'''
#Order modes according to its amplitude (only for presentation purposes): classify them between the components
delta, omega, Phi, bJov = pyLOM.DMD.order_modes(delta, omega, Phi, bJov)


#Plots: separate the different modes according to the components

#Ritz Spectrum
theta = np.array(range(101))*2*np.pi/100
fig1 = plt.figure()
ax  = fig1.add_subplot(111)
ax.plot(np.cos(theta), np.sin(theta), c = 'r')
ax.scatter(muReal, muImag, c = 'b')
ax.axis('equal')
ax.set(xlabel = '\mu_{Re}', ylabel = '\mu_{Imag}', title = 'Ritz spectrum')

#Amplitude vs frequency
fig2 = plt.figure()
ax  = fig2.add_subplot(111)
ax.scatter(omega/(2*np.pi), np.abs(bJov)/np.max(np.abs(bJov)), marker = 'X')
ax.set(xlabel = 'f [Hz]', ylabel = 'Amplitude', title = 'Amplitude vs Frequency of the DMD Modes')
ax.set_yscale('log')

#Damping ratio vs frequency
fig3 = plt.figure()
ax  = fig3.add_subplot(111)
ax.scatter(omega/(2*np.pi), np.abs(delta), marker = 'X')
ax.set(xlabel = 'f [Hz]', ylabel = 'Damping ratio', title = 'Damping ratio vs Frequency of the DMD Modes')
ax.set_yscale('log')

#Scaled amplitude with the damping ratio vs frequency
scaledAmp = np.abs(bJov)*(np.exp(delta*dt) - 1)/delta
fig4 = plt.figure()
ax  = fig4.add_subplot(111)
ax.scatter(omega/(2*np.pi), scaledAmp/np.max(scaledAmp), marker = 'X')
ax.set(xlabel = 'f [Hz]', ylabel = 'Scaled amplitude', title = 'Scaled amplitude with damping ratio vs Frequency of the DMD Modes')
ax.set_yscale('log')

pyLOM.plotDMDMode(Phi, d.xyz, d.mesh, omega/(2*np.pi), modes = [1, 2, 3])

## Show and print timings
pyLOM.cr_stop('example',0)
pyLOM.cr_info()
pyLOM.show_plots()
'''
