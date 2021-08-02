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
UALL = np.load('DATA/UALL.npy') # Data must be in C order
X    = UALL
N    = 151


## Compute POD after subtracting mean (i.e., do PCA)
pyLOM.cr_start('example',0)
Uavg = pyLOM.POD.temporal_mean(X)
X_m  = pyLOM.POD.subtract_mean(X,Uavg)
Y    = X_m

PSI,S,V = pyLOM.POD.svd(Y,transpose_v=False)
pyLOM.cr_stop('example',0) # PSI are POD modes

# Plot accumulative S
plt.figure(figsize=(8,6),dpi=100,facecolor='w',edgecolor='k')

#TODO: refer millor
accumulative_S = np.zeros((1,N));
diag_S = np.diag(S);

for i in range(N):
    accumulative_S[0, i] = np.linalg.norm(diag_S[i:N],2)/np.linalg.norm((diag_S),2);
plt.semilogy(np.linspace(1, N, N), np.transpose(accumulative_S), 'bo')
plt.ylabel('varepsilon1')
plt.xlabel('Truncation size')
plt.title('Tolerance')
plt.ylim((0, 1))


## V representation
fig, ax = plt.subplots(3,1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k',gridspec_kw = {'hspace':0.5})

pyLOM.cr_start('example',0)
dt = 0.2;
t  = dt*np.arange(V.shape[1])
m  = 1 # POD temporal mode number
y  = V[m-1,:]
pyLOM.cr_stop('example',0) # PSI are POD modes
ax[1].plot(t,y,'b')
ax[1].set_title('POD temporal mode m=%d'%m)


## Fast Fourier Transform of V
pyLOM.cr_start('example',0)
PSD  = pyLOM.POD.power_spectral_density(y)
freq = 1./dt/y.shape[0]*np.arange(y.shape[0])
L = int(np.floor(V.shape[0]/2))
pyLOM.cr_stop('example',0) # PSI are POD modes
ax[2].plot(freq[:L],PSD[:L])
ax[2].set_title('Power Spectrum')
ax[2].set_xlabel('St')


## Show and print timings
pyLOM.cr_info()
plt.show()
