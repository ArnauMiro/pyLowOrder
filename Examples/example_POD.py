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
DATAFILE = 'DATA/CYLINDER.h5'
mode     = 1


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
X  = d['UALL']
t  = d.time
dt = d.time[1] - d.time[0]


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
ax[1].plot(t[:S.shape[0]],V[0,:],'b')
ax[1].set_title('POD temporal mode m=%d'%mode)
# Plot FFT of V
L = int(np.floor(V.shape[0]/2))
ax[2].plot(freq[:L],PSD[:L])
ax[2].set_title('Power Spectrum')
ax[2].set_xlabel('St')


## Show and print timings
pyLOM.cr_info()
plt.show()
