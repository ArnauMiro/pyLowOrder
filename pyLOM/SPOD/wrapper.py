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

from ..utils.gpu import cp, gpu_to_cpu, cpu_to_gpu
from ..vmmath    import temporal_mean, subtract_mean, tsqr_svd, hammwin
from ..utils     import cr_nvtx as cr, cr_start, cr_stop
from ..utils.errors import raiseError


def _fft(Xf, winWeight, nDFT, nf):
	fft = cp.fft.fft2 if type(Xf) is cp.ndarray else scipy.fft.fft2
	return (winWeight/nDFT)*fft(Xf,axes=(-1,))[:,:nf]


## SPOD run method
@cr('SPOD.run')
def run(X:np.ndarray, t:np.ndarray, nDFT:int=0, nolap:int=0, remove_mean:bool=True, W:np.ndarray=None):
	r'''
	Run SPOD analysis of a matrix X.

	Args:
		X (np.ndarray): data matrix.
		t (np.ndarray): times at which the snapshots of X were collected
		nDFT (int, optional): number of points in each window (0 will set default value: ~10% nt)
		nolap (int, optional): number of overlap points between windows (0 will set default value: 50% nwin)
		remove_mean (bool, optional): whether or not to remove the mean flow (default, ``True``)
		W (np.ndarray, optional): 1D array with the diagonal of the inner-product weight matrix W

	Returns:
		[(np.ndarray), (np.ndarray), (np.ndarray)]: where the first array is L, the modal energy spectra, the second array is  P, SPOD modes, whose spatial dimensions are identical to those of X and finally f is the frequency vectors
	''' 
	cnp = cp if type(X) is cp.ndarray else np
	# Ensure a C-contiguous buffer, as the compiled vmmath routines
	# assume a packed layout (strided views would be silently misread)
	X = cnp.ascontiguousarray(X)
	M,N = X.shape
	dt  = t[1] - t[0]
	cdtype = np.complex128 if X.dtype is np.double else np.complex64

	# Inner-product weight matrix W (given by its diagonal); it is the
	# user's responsibility to provide a valid weight vector
	W_sqrt = None if W is None else cpu_to_gpu(np.sqrt(np.ascontiguousarray(gpu_to_cpu(W),dtype=X.dtype)))
	
	if nDFT == 0:
		nDFT = int(np.power(2,np.floor(np.log2(N/10))))
	window = cpu_to_gpu(hammwin(nDFT))
	if nolap == 0:
		nolap = int(np.floor(nDFT/2))
	nBlks = int(np.floor((N-nolap)/(nDFT-nolap)))
	# The block SVD (TSQR) requires nBlks <= M (spatial points)
	if nBlks > M:
		raiseError('SPOD.run: number of blocks (%d) exceeds the number of spatial points (%d)! Reduce nDFT/nolap to obtain fewer blocks.' % (nBlks,M))
	# Correction for FFT window gain
	winWeight = 1/np.mean(window)

	# Remove temporal mean
	if remove_mean:
		cr_start('SPOD.temporal_mean',0)
		X_mean = temporal_mean(X)
		Y      = subtract_mean(X, X_mean)
		cr_stop('SPOD.temporal_mean',0)
	else:
		Y = X.copy()

	# Energy norm: scale the data by the square root of the diagonal of W
	# so that the plain SVD below yields the modal energies in the W inner product
	if W_sqrt is not None:
		Y = Y*W_sqrt[:,None]

	# Set frequency axis
	f  = cnp.arange(np.ceil(nDFT / 2) + 1) / dt / nDFT
	nf = f.shape[0]
	qk = cnp.zeros((M,nf),cdtype)
	Q  = cnp.zeros((M*nf,nBlks),cdtype)
	cr_start('SPOD.fft',0)
	for iblk in range(nBlks):
		# Get time index for present block
		i0 = iblk*(nDFT - nolap)
		ix = np.arange(nDFT) + i0
		Xf = Y[:,ix].copy()*window
		qk[:, :]    = _fft(Xf, winWeight, nDFT, nf)
		qk[:,1:-1] *= 2
		Q[:, iblk]  = qk.reshape((M*nf), order='F')
	cr_stop('SPOD.fft',0)

	L  = cnp.zeros((nf,nBlks),X.dtype)
	P  = cnp.zeros((M*nBlks,nf),X.dtype)
	cr_start('SPOD.SVD',0)
	for ifreq, freq in enumerate(f):
		qf         = Q[ifreq*M:(ifreq+1)*M, :].copy()/cnp.sqrt(nBlks)
		U, S, V    = tsqr_svd(qf)
		P[:,ifreq] = cnp.real(U.reshape((M*nBlks), order='F'))
		L[ifreq,:] = cnp.abs(S*S)
	cr_stop('SPOD.SVD',0)

	# Recover the physical, W-orthonormal modes from the transformed ones
	if W_sqrt is not None:
		P = P/cnp.tile(W_sqrt,nBlks)[:,None]

	cr_start('SPOD.sort',0)
	order = cnp.argsort(L[:,0])[::-1]
	P = P[:, order]
	f = f[order]
	L = L[order,:]
	cr_stop('SPOD.sort',0)
	  
	return L, P, f
