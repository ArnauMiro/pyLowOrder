#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import pyLOM

## Parameters
mode = 1


## Data loading
UALL = np.load('DATA/UALL.npy') # Data must be in C order
X    = UALL
N    = 151
dt   = 0.2


## Compute POD after subtracting mean (i.e., do PCA)
pyLOM.cr_start('example',0)
# Run POD
PSI,S,V = pyLOM.POD.run(X,r=1e-6) # PSI are POD modes
# Obtain PSD of the first mode
PSD,freq = pyLOM.POD.PSD(V,dt,m=mode) 
pyLOM.cr_stop('example',0)


## Plots
fig, ax = plt.subplots(3,1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k',gridspec_kw = {'hspace':0.5})
# Plot POD temporal mode
t = dt*np.arange(V.shape[1])
ax[1].plot(t,V[0,:],'b')
ax[1].set_title('POD temporal mode m=%d'%mode)
# Plot FFT of V
L = int(np.floor(V.shape[0]/2))
ax[2].plot(freq[:L],PSD[:L])
ax[2].set_title('Power Spectrum')
ax[2].set_xlabel('St')


## Plot accumulative S
#plt.figure(figsize=(8,6),dpi=100)
#
#accumulative_S = np.zeros((1,N));
#diag_S = np.diag(S);
#
#for i in range(N):
#    accumulative_S[0, i] = np.linalg.norm(diag_S[i:N],2)/np.linalg.norm((diag_S),2);
#plt.semilogy(np.linspace(1, N, N), np.transpose(accumulative_S), 'bo')
#plt.ylabel('varepsilon1')
#plt.xlabel('Truncation size')
#plt.title('Tolerance')
#plt.ylim((0, 1))


## Show and print timings
pyLOM.cr_info()
plt.show()
