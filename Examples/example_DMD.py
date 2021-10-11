#!/usr/bin/env python
#
# Example of POD following the MATLAB script.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import pyLOM


## Data loading
d  = pyLOM.Dataset.load('Examples/Data/CYLINDER.h5')
X  = d['UALL']

fig, ax = plt.subplots(3,1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k',gridspec_kw = {'hspace':0.5})


## Compute POD
pyLOM.cr_start('example',0)

#Compute and substract temporal subtract_mean
Xavg = pyLOM.DMD.temporal_mean(X)
X_m  = pyLOM.DMD.subtract_mean(X, Xavg)

# Separate X1 and X2
X1 = X_m[:, :-1]
X2 = X_m[:, 1:]

# Compute SVD
PSI,S,V = pyLOM.DMD.svd(X_m, 0, X1.shape[1])

# Truncate according to residual
N,res = pyLOM.DMD.residual(S,r = 1e-6)
print('Truncating at %d with a residual of %.2e'%(N, res))
PSI = PSI[:, :N]
S   = S[:N]
V   = np.transpose(V[:N, :])

A = np.matmul(np.matmul(np.matmul(np.transpose(PSI), X2), V), np.diag(1/S)) #descomposició xunga que m'ha d'explicar el Beka
Ahat = np.matmul(np.matmul(np.diag(1/np.sqrt(S)), A), np.diag(np.sqrt(S))) #la A de les diapos
delta, omega, w = pyLOM.DMD.eigen(Ahat) #eigenvalues i eigenvectors

Wr  = np.matmul(np.diag(np.sqrt(S)), w) #eigenvectors amb una transformació liada que tmb m'ha d'explicar el Beka
Phi = np.matmul(np.matmul(np.matmul(X2, V), np.diag(1/S)), Wr) #DMD modes
print(Phi.shape)
print(Phi)
pene
'''
TODO:
    * Compute lambda (he de parlar amb el Beka)
    * Plots que hi ha al MATLAB
    * Reconstrucció
    * Distribuir això en funcions de cython
    * Implementar el producte de matrius amb una altra llibreria?
    * Preparar i correr els altres exemples amb les funcions bones
'''

pyLOM.cr_stop('example',0)

## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()
