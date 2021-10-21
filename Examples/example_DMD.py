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
X  = d['UALL']
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
Atilde = np.matmul(np.matmul(np.matmul(np.transpose(PSI), X2), np.transpose(V)), np.diag(1/S))

#Eigendecomposition of Atilde
muReal, muImag, w = pyLOM.DMD.eigen(Atilde)
#Creation of a matrix containing the complex numbers of the eigenvectors
wComplex = np.zeros(w.shape, dtype = 'complex_')
ivec = 0
while ivec < w.shape[1] - 1:
    if muImag[ivec] > np.finfo(np.double).eps:
        wComplex[:, ivec]     = w[:, ivec] + w[:, ivec + 1]*1j
        wComplex[:, ivec + 1] = w[:, ivec] - w[:, ivec + 1]*1j
        ivec += 2
    else:
        wComplex[:, ivec] = w[:, ivec] + 0*1j
        ivec = ivec + 1
#Compute modulus and argument of the eigenvalues
muModulus = np.sqrt(muReal*muReal + muImag*muImag)
muArg     = np.arctan2(muImag, muReal)

#Computation of the damping ratio of the mode
delta = np.log(muModulus)/dt

#Computation of the frequency of the mode
omega = muArg/dt

#Computation of the modes
Phi = np.matmul(np.matmul(np.matmul(X2, np.transpose(V)), np.diag(1/S)), wComplex)

#Computation of the amplitudes according to Jovanovic 2014
#Creation of the Vandermonde matrix
Vand  = np.zeros((muReal.shape[0], X1.shape[1]), dtype = 'complex_')
for icol in range(X1.shape[1]):
    VandModulus   = muModulus**icol
    VandArg       = muArg*icol
    Vand[:, icol] = VandModulus*np.cos(VandArg) + VandModulus*np.sin(VandArg)*1j
#Compute the amplitudes
P    = np.matmul(np.transpose(np.conj(wComplex)), wComplex)*np.conj(np.matmul(Vand, np.transpose(np.conj(Vand))))
Pl   = np.linalg.cholesky(P)
G    = np.matmul(np.diag(S), V)
q    = np.conj(np.diag(np.matmul(np.matmul(Vand, np.transpose(np.conj(G))), wComplex)))
bJov = np.matmul(np.linalg.inv(np.transpose(np.conj(Pl))), np.matmul(np.linalg.inv(Pl), q)) #Amplitudes according to Jovanovic 2014

#Reconstruction according to Jovanovic 2014
Xdmd = np.matmul(np.matmul(np.matmul(PSI, wComplex), np.diag(bJov)), Vand)

#Order modes according to its amplitude
dummy1 = np.concatenate((bJov, delta, omega, Phi), axis = 1)
dummy2 = dummy1[np.argsort(dummy1, 0, axis = 0)]

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

pyLOM.plotDMDMode(Phi, d.xyz, d.mesh, modes = [1, 2])
'''
TODO:
    * Plots que hi ha al MATLAB
    * Distribuir això en funcions de cython
    * Preparar i correr els altres exemples amb les funcions bones
'''


## Show and print timings
pyLOM.cr_stop('example',0)
pyLOM.cr_info()
pyLOM.show_plots()
