#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for SPOD.
#
# Last rev: 02/09/2022
from __future__ import print_function

import numpy as np
import scipy

from ..vmmath       import temporal_mean, subtract_mean, tsqr_svd
from ..utils.cr     import cr, cr_start, cr_stop
from ..utils.errors import raiseError


def _hammwin(N):
	return np.transpose(0.54-0.46*np.cos(2*np.pi*np.arange(N)/(N-1)))

def _fft(Xf, winWeight, nDFT, nf):
	return (winWeight/nDFT)*scipy.fft.fft(Xf)[:nf]


## SPOD run method
@cr('SPOD.run')
def run(X, t, nDFT=0, nolap=0, remove_mean=True):
	'''
	Run SPOD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,nt]: data matrix
		- dt:                timestep between adjacent snapshots
		- npwin:             number of points in each window (0 will set default value: ~10% nt)
		- nolap:             number of overlap points between windows (0 will set default value: 50% nwin)
		- remove_mean:       whether or not to remove the mean flow

	Returns:
		- L:  modal energy spectra.
		- P:  SPOD modes, whose spatial dimensions are identical to those of X.
		- f:  frequency vector.
	''' 
	M = X.shape[0]
	N = X.shape[1]
	dt = t[1] - t[0]
	
	if nDFT == 0:
		nDFT = int(np.power(2,np.floor(np.log2(N/10))))
	window = _hammwin(nDFT)
	if nolap == 0:
		nolap = int(np.floor(nDFT/2))
	nBlks = int(np.floor((N-nolap)/(nDFT-nolap)))
	#Correction for FFT window gain
	winWeight = 1/np.mean(window)

	#Remove temporal mean
	if remove_mean:
		cr_start('SPOD.temporal_mean',0)
		X_mean = temporal_mean(X)
		Y      = subtract_mean(X, X_mean)
		cr_stop('SPOD.temporal_mean',0)
	else:
		Y = X.copy()

	#Set frequency axis
	f  = np.arange(np.ceil(nDFT / 2) + 1) / dt / nDFT
	nf = f.shape[0]
	qk = np.zeros((M,nf),np.complex128)
	Q  = np.zeros((M*nf,nBlks),np.complex128)
	L  = np.zeros((nf,nBlks),np.double)
	P  = np.zeros((M*nBlks,nf),np.double)
	cr_start('SPOD.fft',0)
	for iblk in range(nBlks):
		# Get time index for present block
		i0 = iblk*(nDFT - nolap)
		ix = np.arange(nDFT) + i0
		for ip in range(M):
			Xf = Y[ip, ix].copy()*window
			qk[ip, :] = _fft(Xf, winWeight, nDFT, nf)
		qk[:,1:-1] *= 2
		Q[:, iblk] = qk.reshape((M*nf), order='F')
	cr_stop('SPOD.fft',0)

	cr_start('SPOD.SVD',0)
	for ifreq, freq in enumerate(f):
		qf         = Q[ifreq*M:(ifreq+1)*M, :].copy()/np.sqrt(nBlks)
		U, S, V    = tsqr_svd(qf)
		P[:,ifreq] = np.real(U.reshape((M*nBlks), order='F'))
		L[ifreq,:] = np.abs(S*S)
	cr_stop('SPOD.SVD',0)

	cr_start('SPOD.sort',0)
	order = np.argsort(L[:,0])[::-1]
	P = P[:, order]
	f = f[order]
	L = L[order,:]
	cr_stop('SPOD.sort',0)
	  
	return L, P, f
