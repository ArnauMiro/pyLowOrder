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
d  = pyLOM.Dataset.load('DATA/CYLINDER.h5')
X  = d['UALL']

fig, ax = plt.subplots(3,1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k',gridspec_kw = {'hspace':0.5})


## Compute POD
pyLOM.cr_start('example',0)
# Compute and substract temporal mean
Uavg = pyLOM.POD.temporal_mean(X)
X_m  = pyLOM.POD.subtract_mean(X,Uavg)
Y    = X_m

# Compute SVD
PSI,S,V = pyLOM.POD.svd(Y)

# Plot accumulative S
pyLOM.plotResidual(S)

# Truncate according to residual
N,res = pyLOM.POD.residual(S,r=5e-6)
print('POD: truncating at %d with a residual of %.2e'%(N,res))
PSI = PSI[:,:N]
S   = S[:N]
V   = V[:N,:]

# V representation
dt = 0.2;
t  = dt*np.arange(V.shape[1])
m  = 1 # POD temporal mode number
y  = V[m-1,:]
ax[1].plot(t,y,'b')
ax[1].set_title('POD temporal mode m=%d'%m)

# Fast Fourier Transform of V
PSD  = pyLOM.POD.power_spectral_density(y)
freq = 1./dt/y.shape[0]*np.arange(y.shape[0])
L    = int(np.floor(V.shape[0]/2))
ax[2].plot(freq[:L],PSD[:L])
ax[2].set_title('Power Spectrum')
ax[2].set_xlabel('St')

# Reconstruction
X_POD = np.matmul(PSI,np.matmul(S*np.identity(S.shape[0],np.double),V))

pyLOM.cr_stop('example',0)


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()