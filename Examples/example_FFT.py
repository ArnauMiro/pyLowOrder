#!/usr/bin/env python
#
# Example how to perform FFT analysis.
#
# Last revision: 21/06/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import pyLOM


## Parameters
Fs = 1000.  # Sampling frequency
T  = 1/Fs   # Sampling period
L  = 1500   # Number of samples


## Build a noisy signal
t = np.linspace(0,np.pi/2,L)#np.arange(L)*T
S = 0.7*np.sin(50*2.*np.pi*t) + np.sin(120*2.*np.pi*t)
X = S + 2*np.random.randn(t.size)


## Compute FFT in different ways
f1,p1 = pyLOM.math.fft(t,X)
f2,p2 = pyLOM.math.fft(t,X,equispaced=False)


## Plot
fig, ax = plt.subplots(3,1,figsize=(8,12),dpi=100,facecolor='w',edgecolor='k',gridspec_kw={'hspace':0.25,'wspace':0.25})

# Signal plot
ax[0].plot(1000*t,X,'k')
ax[0].set_xlim([0,50])
ax[0].set_xlabel('time [msec]')
ax[0].set_ylabel('Y(t)')

# FFT spectrum plot
ax[1].plot(f1,p1,'k')
ax[1].set_xlim([0,150])
ax[1].set_xlabel('f [Hz]')
ax[1].set_ylabel('|P1(f)|')

# FFT spectrum plot
ax[2].plot(f2,p2,'k')
ax[2].set_xlim([0,150])
ax[2].set_xlabel('f [Hz]')
ax[2].set_ylabel('|P1(f)|')


pyLOM.cr_info()
plt.show()