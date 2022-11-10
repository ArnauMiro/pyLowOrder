#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np
import pyLOM
import pyAlya
import matplotlib.pyplot as plt
from scipy.io import loadmat

'''
Run SPOD analysis of a matrix X.
Inputs:
    - X[ndims*nmesh,nt]: data matrix
    - dt:                timestep between adjacent snapshots
    - npwin:             number of points in each window (0 will set default value: ~10% nt)
    - nolap:             number of overlap points between windows (0 will set default value: 50% nwin)
    - weight:            weight function for each grid point (1 will set unitary weight for all windows)
    - window:            give values to the window function
    - remove_mean:       whether or not to remove the mean flow
Returns:
    - L:  modal energy spectra.
    - P:  SPOD modes, whose spatial dimensions are identical to those of X.
    - F:  frequency vector.
'''

## Data loading
DATAFILE = '../UPM_BSC_LowOrder/Examples/Data/CYLINDER.h5'
VARIABLE = 'VELOX'
#beka     = loadmat('../SPOD_DEMO/spod_re100_results.mat')
d        = pyLOM.Dataset.load(DATAFILE)
X        = d[VARIABLE]
M        = X.shape[0]
N        = X.shape[1]
t        = d.time
dt       = t[1] - t[0]

## SPOD Input data
npwin       = 60
nolap       = 0
weight      = np.ones(M)
remove_mean = True

## Run SPOD
## Parse input data to compute the number of blocks (ES POT POSAR EN UNA ALTRA FUNCIÓ)
# Number of points per window
if npwin == 0:
    npwin = int(np.floor(0.1*N))
    npwin = int(2**(np.floor(np.log2(npwin)))) # round-up to 2**n type int
else:
    npwin = int(npwin)
    #inpwin = int(2**(np.floor(np.log2(npwin)))) # round-up to 2**n type int
# Number of overlapping snapshots
if nolap == 0:
    nolap = int(np.floor(0.5*npwin))
else:
    if nolap > npwin -1:
        pyLOM.raiseError('Number of overlapping snapshots is too large')
#Compute number of blocks
nBlks = int(np.floor((N-nolap)/(npwin-nolap)))

## Loop over the blocks
nf = np.int(np.floor(npwin/2)+1)
F  = np.zeros((M, nf))
Q  = np.zeros((M*nf, nBlks))
for iblk in range(nBlks):
    # Get time index for present block
    it_start = iblk*(npwin - nolap)
    it_end   = it_start + npwin
    for ip in range(M):
        Xf       = X[ip, it_start:it_end].copy()
        f, s     = pyAlya.postproc.fft_periodogram(t[it_start:it_end], Xf)
        F[ip, :] = s
    Q[:, iblk] = F.reshape((M*nf), order='C')

L = np.zeros((nf, nBlks))
P = np.zeros((M*nf, nBlks))
for ifreq in range(nf):
    Qf = Q[ifreq*M:(ifreq+1)*M,:]
    Qr  = pyLOM.math.matmul(np.conjugate(np.transpose(Qf)), Qf)
    mur, mui, w = pyLOM.math.eigen(Qr)
    L[ifreq, :] = np.sqrt(mur*mur + mui*mui)
    P[ifreq*M:(ifreq+1)*M, :] = pyLOM.math.matmul(pyLOM.math.matmul(Qf, w), np.diag(1/np.sqrt(nBlks*(mur+mui*1j))))

#for iblk in range(nBlks):
#    plt.loglog(f, beka['L'][:,iblk], '*', label = iblk)
#    plt.loglog(f, L[:,iblk], 'o', label = iblk)
plt.legend()
plt.show()