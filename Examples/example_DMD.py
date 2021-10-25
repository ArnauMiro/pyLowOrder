#!/usr/bin/env python
#
# Example of POD following the MATLAB script.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

import pyLOM

## Data loading
d  = pyLOM.Dataset.load('Examples/Data/CYLINDER.h5')
X  = d['VELOC']
X  = X[:89351, :].copy()
dt = 0.2

#fig, ax = plt.subplots(3,1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k',gridspec_kw = {'hspace':0.5})
pyLOM.cr_start('example',0)

#Compute and substract temporal subtract_mean
Xavg = pyLOM.DMD.temporal_mean(X)
X_m  = pyLOM.DMD.subtract_mean(X, Xavg)

# Separate X1 and X2
X1, X2 = pyLOM.DMD.matrix_split(X_m)

# Compute SVD
PSI,S,V = pyLOM.DMD.svd(X_m, 0, X1.shape[1])

# Truncate according to residual
PSI, S, V = pyLOM.DMD.truncate(PSI, S, V, r = 1e-6)

#Project A (Jacobian of the snapshots) into POD basis
Atilde = pyLOM.DMD.project_POD_basis(PSI, X2, V, S)

#Eigendecomposition of Atilde
muReal, muImag, w = pyLOM.DMD.eigen(Atilde)
#Creation of a matrix containing the complex numbers of the eigenvectors
wComplex = pyLOM.DMD.build_complex_eigenvectors(w, muImag)

#Compute frequency and damping ratio of the modes
delta, omega, muModulus, muArg = pyLOM.DMD.frequency_damping(muReal, muImag, dt)

#Computation of the modes
Phi = pyLOM.DMD.mode_computation(X2, V, S, wComplex)

#Computation of the amplitudes according to Jovanovic 2014
Vand = pyLOM.DMD.vandermonde(muReal, muImag, muReal.shape[0], X1.shape[1])
bJov = pyLOM.DMD.amplitude_jovanovic(muReal, muImag, muReal.shape[0], X1.shape[1], wComplex, S, V, Vand)

#Reconstruction according to Jovanovic 2014
Xdmd = np.matmul(np.matmul(np.matmul(PSI, wComplex), np.diag(bJov)), Vand)

#Order modes according to its amplitude (only for presentation purposes)
delta, omega, Phi, bJov = pyLOM.DMD.order_modes(delta, omega, Phi, bJov)


#Plots

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
