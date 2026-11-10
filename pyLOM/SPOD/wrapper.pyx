#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for SPOD.
#
# Last rev: 02/09/2022
from __future__ import print_function

cimport cython
cimport numpy as np

import numpy as np

#from libc.complex  cimport creal, cimag
cdef extern from "<complex.h>" nogil:
	float  complex I
	# Decomposing complex values
	float cimagf(float complex z)
	float crealf(float complex z)
	double cimag(double complex z)
	double creal(double complex z)
cdef double complex J = 1j
from libc.stdlib     cimport malloc, free
from libc.string     cimport memcpy, memset
from libc.math       cimport pow, floor, ceil, log2, sqrt
from ..vmmath.cfuncs cimport real, real_complex
from ..vmmath.cfuncs cimport c_ssort, c_svector_mean, c_stemporal_mean, c_ssubtract_mean, c_shammwin, c_sfft1D, c_ctsqr_svd
from ..vmmath.cfuncs cimport c_dsort, c_dvector_mean, c_dtemporal_mean, c_dsubtract_mean, c_dhammwin, c_dfft1D, c_ztsqr_svd

from ..utils.cr       import cr, cr_start, cr_stop
from ..utils.errors   import raiseError


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _sfft(np.complex64_t *out, float *Xf, float winWeight, int nDFT, int nf):
	cdef int i
	cdef float fact = winWeight/nDFT
	cdef np.complex64_t *Yf
	# Allocate memory
	Yf = <np.complex64_t*>malloc(nDFT*sizeof(np.complex64_t))
	# Compute FFT
	c_sfft1D(Yf,Xf,nDFT)
	# Multiply by a factor
	for i in range(nf):
		out[i] = fact*Yf[i]
	# Free memory
	free(Yf)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _dfft(np.complex128_t *out, double *Xf, double winWeight, int nDFT, int nf):
	cdef int i
	cdef double fact = winWeight/nDFT
	cdef np.complex128_t *Yf
	# Allocate memory
	Yf = <np.complex128_t*>malloc(nDFT*sizeof(np.complex128_t))
	# Compute FFT
	c_dfft1D(Yf,Xf,nDFT)
	# Multiply by a factor
	for i in range(nf):
		out[i] = fact*Yf[i]
	# Free memory
	free(Yf)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _ssort(float *L, float *P, float *f, int M, int nBlks, int nf):
	cdef int ii, jj, r, i, j, k, icyc, ncyc
	cdef float *v
	cdef int *sigma
	cdef int *start
	cdef float *tmpl
	cdef float tmp, tmp2
	v     = <float*>malloc(nf*sizeof(float))
	sigma = <int*>malloc(nf*sizeof(int))
	start = <int*>malloc(nf*sizeof(int))
	tmpl  = <float*>malloc(nBlks*sizeof(float))
	# Copy to v
	for ii in range(nf):
		v[ii] = L[nBlks*ii + 0]
	# Obtain index array that sorts v
	c_ssort(v,sigma,nf)
	free(v)
	# Build the destination permutation sigma: element ii goes to
	# position nf-sigma[ii]-1 (reversed order, i.e. descending)
	for ii in range(nf):
		sigma[ii] = nf-sigma[ii]-1
	# Decompose sigma into cycles, recording the starting element of each
	# cycle in start (elements already visited are marked negative).
	# Cycles of length one need no reordering and are skipped.
	ncyc = 0
	for i in range(nf):
		if sigma[i] < 0: continue # already in a previous cycle
		if sigma[i] == i: # cycle of length one: mark and skip
			sigma[i] = -i-1
			continue
		start[ncyc] = i
		ncyc = ncyc + 1
		j = i
		while True:
			k        = sigma[j]
			sigma[j] = -k-1 # mark as visited
			if k == i: break # cycle closed
			j = k
	# Restore sigma
	for ii in range(nf):
		sigma[ii] = -sigma[ii]-1
	# Reorder f and L in place: rotate each cycle using a single temporary
	# element (one row of nBlks elements for L), so that no copy of f, L
	# or P is needed. Element j of the cycle goes to position sigma[j].
	for icyc in range(ncyc):
		i   = start[icyc]
		j   = i
		tmp = f[i]
		memcpy(tmpl,&L[nBlks*i],nBlks*sizeof(float))
		while True:
			k = sigma[j]
			if k == i: break # cycle closed
			# Rotate f
			tmp2 = f[k]; f[k] = tmp; tmp = tmp2
			# Rotate L
			for jj in range(nBlks):
				tmp2          = L[nBlks*k+jj]
				L[nBlks*k+jj] = tmpl[jj]
				tmpl[jj]      = tmp2
			j = k
		f[i] = tmp
		memcpy(&L[nBlks*i],tmpl,nBlks*sizeof(float))
	# Reorder P in place: the same permutation acts on the columns of each
	# of the M*nBlks rows of P (contiguous arrays of nf elements), which
	# are rotated cycle by cycle using a single temporary element
	for r in range(M*nBlks):
		for icyc in range(ncyc):
			i   = start[icyc]
			j   = i
			tmp = P[nf*r+i]
			while True:
				k = sigma[j]
				if k == i: break # cycle closed
				tmp2      = P[nf*r+k]
				P[nf*r+k] = tmp
				tmp       = tmp2
				j         = k
			P[nf*r+i] = tmp
	free(start)
	free(tmpl)
	free(sigma)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _dsort(double *L, double *P, double *f, int M, int nBlks, int nf):
	cdef int ii, jj, r, i, j, k, icyc, ncyc
	cdef double *v
	cdef int *sigma
	cdef int *start
	cdef double *tmpl
	cdef double tmp, tmp2
	v     = <double*>malloc(nf*sizeof(double))
	sigma = <int*>malloc(nf*sizeof(int))
	start = <int*>malloc(nf*sizeof(int))
	tmpl  = <double*>malloc(nBlks*sizeof(double))
	# Copy to v
	for ii in range(nf):
		v[ii] = L[nBlks*ii + 0]
	# Obtain index array that sorts v
	c_dsort(v,sigma,nf)
	free(v)
	# Build the destination permutation sigma: element ii goes to
	# position nf-sigma[ii]-1 (reversed order, i.e. descending)
	for ii in range(nf):
		sigma[ii] = nf-sigma[ii]-1
	# Decompose sigma into cycles, recording the starting element of each
	# cycle in start (elements already visited are marked negative).
	# Cycles of length one need no reordering and are skipped.
	ncyc = 0
	for i in range(nf):
		if sigma[i] < 0: continue # already in a previous cycle
		if sigma[i] == i: # cycle of length one: mark and skip
			sigma[i] = -i-1
			continue
		start[ncyc] = i
		ncyc = ncyc + 1
		j = i
		while True:
			k        = sigma[j]
			sigma[j] = -k-1 # mark as visited
			if k == i: break # cycle closed
			j = k
	# Restore sigma
	for ii in range(nf):
		sigma[ii] = -sigma[ii]-1
	# Reorder f and L in place: rotate each cycle using a single temporary
	# element (one row of nBlks elements for L), so that no copy of f, L
	# or P is needed. Element j of the cycle goes to position sigma[j].
	for icyc in range(ncyc):
		i   = start[icyc]
		j   = i
		tmp = f[i]
		memcpy(tmpl,&L[nBlks*i],nBlks*sizeof(double))
		while True:
			k = sigma[j]
			if k == i: break # cycle closed
			# Rotate f
			tmp2 = f[k]; f[k] = tmp; tmp = tmp2
			# Rotate L
			for jj in range(nBlks):
				tmp2          = L[nBlks*k+jj]
				L[nBlks*k+jj] = tmpl[jj]
				tmpl[jj]      = tmp2
			j = k
		f[i] = tmp
		memcpy(&L[nBlks*i],tmpl,nBlks*sizeof(double))
	# Reorder P in place: the same permutation acts on the columns of each
	# of the M*nBlks rows of P (contiguous arrays of nf elements), which
	# are rotated cycle by cycle using a single temporary element
	for r in range(M*nBlks):
		for icyc in range(ncyc):
			i   = start[icyc]
			j   = i
			tmp = P[nf*r+i]
			while True:
				k = sigma[j]
				if k == i: break # cycle closed
				tmp2      = P[nf*r+k]
				P[nf*r+k] = tmp
				tmp       = tmp2
				j         = k
			P[nf*r+i] = tmp
	free(start)
	free(tmpl)
	free(sigma)


## SPOD run method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _srun(float[:,:] X, float[:] t, int nDFT, int nolap, int remove_mean, float[:] W):
	'''
	Run SPOD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,nt]: data matrix
		- dt:                timestep between adjacent snapshots
		- npwin:             number of points in each window (0 will set default value: ~10% nt)
		- nolap:             number of overlap points between windows (0 will set default value: 50% nwin)
		- remove_mean:       whether or not to remove the mean flow
		- W[ndims*nmesh]:    diagonal of the inner-product weight matrix (energy norm); None for no weighting

	Returns:
		- L:  modal energy spectra.
		- P:  SPOD modes, whose spatial dimensions are identical to those of X.
		- f:  frequency vector.
	''' 
	cdef int i, iblk, ifreq, ip, i0, nBlks, nf, M = X.shape[0], N = X.shape[1]
	cdef float winWeight, dt = t[1] - t[0]

	cdef float *window
	cdef float *X_mean
	cdef float *Xf
	cdef float *Y
	cdef float *S
	cdef np.complex64_t *qk
	cdef np.complex64_t *Q
	cdef np.complex64_t *U
	cdef np.complex64_t *V
	cdef np.complex64_t *qf
	cdef bint weighted
	cdef float *W_sqrt

	# Output arrays
	cdef np.ndarray[np.float32_t,ndim=2] L
	cdef np.ndarray[np.float32_t,ndim=2] P
	cdef np.ndarray[np.float32_t,ndim=1] f

	# Deal with the window gain
	if nDFT == 0: 
		nDFT = <int>(pow(2,floor(log2(N/10))))
	# Allocate vector
	window = <float*>malloc(nDFT*sizeof(float))
	# Compute Hamming window
	c_shammwin(window,nDFT)

	# Correction for FFT window gain
	winWeight = 1.0/c_svector_mean(window,0,nDFT)

	if nolap == 0:
		nolap = <int>(floor(nDFT/2))

	nBlks = <int>(floor((N-nolap)/(nDFT-nolap)))

	# The block SVD (TSQR) requires nBlks <= M (spatial points)
	if nBlks > M:
		raiseError('SPOD.run: number of blocks (%d) exceeds the number of spatial points (%d)! Reduce nDFT/nolap to obtain fewer blocks.' % (nBlks,M))

	# Remove temporal mean
	Y = <float*>malloc(M*N*sizeof(float))
	if remove_mean:
		cr_start('SPOD.temporal_mean',0)
		X_mean = <float*>malloc(M*sizeof(float))
		# Compute temporal mean
		c_stemporal_mean(X_mean,&X[0,0],M,N)
		# Compute substract temporal mean
		c_ssubtract_mean(Y,&X[0,0],X_mean,M,N)
		free(X_mean)
		cr_stop('SPOD.temporal_mean',0)
	else:
		memcpy(Y,&X[0,0],M*N*sizeof(float))

	# Inner-product weight (energy norm): diagonal of the weight matrix W.
	# The data is scaled by the square root of the diagonal of W so that the
	# plain SVD below yields the modal energies in the W inner product.
	# Providing a valid weight vector is the user's responsibility.
	weighted = W is not None
	W_sqrt = NULL
	if weighted:
		# Square root of the diagonal of W
		W_sqrt = <float*>malloc(M*sizeof(float))
		for i in range(M):
			W_sqrt[i] = sqrt(W[i])
		# Scale the data rows
		for i in range(M):
			for j in range(N):
				Y[N*i + j] = Y[N*i + j]*W_sqrt[i]

	# Set frequency axis
	nf = <int>(ceil(nDFT/2)) + 1

	f = np.zeros((nf,)       ,dtype=np.float32)
	L = np.zeros((nf,nBlks)  ,dtype=np.float32)
	P = np.zeros((M*nBlks,nf),dtype=np.float32)

	# Set frequency axis
	for i in range(nf):
		f[i] = <float>(i)/dt/<float>(nDFT)

	# Allocate memory
	Xf = <float*>malloc(nDFT*sizeof(float))
	qk = <np.complex64_t*>malloc(M*nf*sizeof(np.complex64_t))
	Q  = <np.complex64_t*>malloc(M*nf*nBlks*sizeof(np.complex64_t))
	cr_start('SPOD.fft',0)
	for iblk in range(nBlks):
		i0 = iblk*(nDFT - nolap)
		for ip in range(M):
			# Populate Xf
			for i in range(nDFT):
				Xf[i] = Y[N*ip + (i + i0)] * window[i]
			# FFT on Xf
			_sfft(&qk[nf*ip],Xf,winWeight,nDFT,nf)
			# Multiply qk
			for i in range(1,nf-1):
				qk[nf*ip + i] *= <float>(2.0)
		# Populate Q
		for ip in range(M):
			for i in range(nf):
				Q[nf*nBlks*ip + nBlks*i + iblk] = qk[nf*ip + i]
	cr_stop('SPOD.fft',0)

	free(qk)
	free(window)
	free(Xf)

	# Allocate memory
	qf = <np.complex64_t*>malloc(M*nBlks*sizeof(np.complex64_t))
	U  = <np.complex64_t*>malloc(M*nBlks*sizeof(np.complex64_t))
	S  = <float*>malloc(nBlks*sizeof(float))
	V  = <np.complex64_t*>malloc(nBlks*nBlks*sizeof(np.complex64_t))

	cr_start('SPOD.SVD',0)
	for ifreq in range(nf):
		# Load block in qf
		for i in range(M):
			for iblk in range(nBlks):
				qf[nBlks*i + iblk] = Q[nf*nBlks*i + nBlks*ifreq + iblk]/sqrt(nBlks)
		# Run SVD
		c_ctsqr_svd(U,S,V,qf,M,nBlks)
		# Store P
		for i in range(M):
			for iblk in range(nBlks):
				P[i + M*iblk,ifreq] = crealf(U[nBlks*i + iblk])
		# Store L
		for iblk in range(nBlks):
			L[ifreq,iblk] = S[iblk]*S[iblk]
	cr_stop('SPOD.SVD',0)

	free(qf)
	free(Q)
	free(U)
	free(S)
	free(V)

	# Recover the physical, W-orthonormal modes from the transformed ones
	if weighted:
		for i in range(M):
			for iblk in range(nBlks):
				for ifreq in range(nf):
					P[i + M*iblk,ifreq] = P[i + M*iblk,ifreq]/W_sqrt[i]
	free(W_sqrt)

	# Sort
	cr_start('SPOD.sort',0)
	_ssort(&L[0,0],&P[0,0],&f[0],M,nBlks,nf)
	cr_stop('SPOD.sort',0)

	return L, P, f

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _drun(double[:,:] X, double[:] t, int nDFT, int nolap, int remove_mean, double[:] W):
	'''
	Run SPOD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,nt]: data matrix
		- dt:                timestep between adjacent snapshots
		- npwin:             number of points in each window (0 will set default value: ~10% nt)
		- nolap:             number of overlap points between windows (0 will set default value: 50% nwin)
		- remove_mean:       whether or not to remove the mean flow
		- W[ndims*nmesh]:    diagonal of the inner-product weight matrix (energy norm); None for no weighting

	Returns:
		- L:  modal energy spectra.
		- P:  SPOD modes, whose spatial dimensions are identical to those of X.
		- f:  frequency vector.
	''' 
	cdef int i, iblk, ifreq, ip, i0, nBlks, nf, M = X.shape[0], N = X.shape[1]
	cdef double winWeight, dt = t[1] - t[0]

	cdef double *window
	cdef double *X_mean
	cdef double *Xf
	cdef double *Y
	cdef np.complex128_t *qk
	cdef np.complex128_t *Q
	cdef np.complex128_t *U
	cdef double *S
	cdef np.complex128_t *V
	cdef np.complex128_t *qf
	cdef bint weighted
	cdef double *W_sqrt

	# Output arrays
	cdef np.ndarray[np.double_t,ndim=2] L
	cdef np.ndarray[np.double_t,ndim=2] P
	cdef np.ndarray[np.double_t,ndim=1] f

	# Deal with the window gain
	if nDFT == 0: 
		nDFT = <int>(pow(2,floor(log2(N/10))))
	# Allocate vector
	window = <double*>malloc(nDFT*sizeof(double))
	# Compute Hamming window
	c_dhammwin(window,nDFT)

	# Correction for FFT window gain
	winWeight = 1.0/c_dvector_mean(window,0,nDFT)

	if nolap == 0:
		nolap = <int>(floor(nDFT/2))

	nBlks = <int>(floor((N-nolap)/(nDFT-nolap)))

	# The block SVD (TSQR) requires nBlks <= M (spatial points)
	if nBlks > M:
		raiseError('SPOD.run: number of blocks (%d) exceeds the number of spatial points (%d)! Reduce nDFT/nolap to obtain fewer blocks.' % (nBlks,M))

	# Remove temporal mean
	Y = <double*>malloc(M*N*sizeof(double))
	if remove_mean:
		cr_start('SPOD.temporal_mean',0)
		X_mean = <double*>malloc(M*sizeof(double))
		# Compute temporal mean
		c_dtemporal_mean(X_mean,&X[0,0],M,N)
		# Compute substract temporal mean
		c_dsubtract_mean(Y,&X[0,0],X_mean,M,N)
		free(X_mean)
		cr_stop('SPOD.temporal_mean',0)
	else:
		memcpy(Y,&X[0,0],M*N*sizeof(double))

	# Inner-product weight (energy norm): diagonal of the weight matrix W.
	# The data is scaled by the square root of the diagonal of W so that the
	# plain SVD below yields the modal energies in the W inner product.
	# Providing a valid weight vector is the user's responsibility.
	weighted = W is not None
	W_sqrt = NULL
	if weighted:
		# Square root of the diagonal of W
		W_sqrt = <double*>malloc(M*sizeof(double))
		for i in range(M):
			W_sqrt[i] = sqrt(W[i])
		# Scale the data rows
		for i in range(M):
			for j in range(N):
				Y[N*i + j] = Y[N*i + j]*W_sqrt[i]

	# Set frequency axis
	nf = <int>(ceil(nDFT/2)) + 1

	f = np.zeros((nf,)       ,dtype=np.double)
	L = np.zeros((nf,nBlks)  ,dtype=np.double)
	P = np.zeros((M*nBlks,nf),dtype=np.double)

	# Set frequency axis
	for i in range(nf):
		f[i] = <double>(i)/dt/<double>(nDFT)

	# Allocate memory
	Xf = <double*>malloc(nDFT*sizeof(double))
	qk = <np.complex128_t*>malloc(M*nf*sizeof(np.complex128_t))
	Q  = <np.complex128_t*>malloc(M*nf*nBlks*sizeof(np.complex128_t))
	cr_start('SPOD.fft',0)
	for iblk in range(nBlks):
		i0 = iblk*(nDFT - nolap)
		for ip in range(M):
			# Populate Xf
			for i in range(nDFT):
				Xf[i] = Y[N*ip + (i + i0)] * window[i]
			# FFT on Xf
			_dfft(&qk[nf*ip],Xf,winWeight,nDFT,nf)
			# Multiply qk
			for i in range(1,nf-1):
				qk[nf*ip + i] *= 2.0
		# Populate Q
		for ip in range(M):
			for i in range(nf):
				Q[nf*nBlks*ip + nBlks*i + iblk] = qk[nf*ip + i]
	cr_stop('SPOD.fft',0)

	free(qk)
	free(window)
	free(Xf)

	# Allocate memory
	qf = <np.complex128_t*>malloc(M*nBlks*sizeof(np.complex128_t))
	U  = <np.complex128_t*>malloc(M*nBlks*sizeof(np.complex128_t))
	S  = <double*>malloc(nBlks*sizeof(double))
	V  = <np.complex128_t*>malloc(nBlks*nBlks*sizeof(np.complex128_t))

	cr_start('SPOD.SVD',0)
	for ifreq in range(nf):
		# Load block in qf
		for i in range(M):
			for iblk in range(nBlks):
				qf[nBlks*i + iblk] = Q[nf*nBlks*i + nBlks*ifreq + iblk]/sqrt(nBlks)
		# Run SVD
		c_ztsqr_svd(U,S,V,qf,M,nBlks)
		# Store P
		for i in range(M):
			for iblk in range(nBlks):
				P[i + M*iblk,ifreq] = creal(U[nBlks*i + iblk])
		# Store L
		for iblk in range(nBlks):
			L[ifreq,iblk] = S[iblk]*S[iblk]
	cr_stop('SPOD.SVD',0)

	free(qf)
	free(Q)
	free(U)
	free(S)
	free(V)

	# Recover the physical, W-orthonormal modes from the transformed ones
	if weighted:
		for i in range(M):
			for iblk in range(nBlks):
				for ifreq in range(nf):
					P[i + M*iblk,ifreq] = P[i + M*iblk,ifreq]/W_sqrt[i]
	free(W_sqrt)

	# Sort
	cr_start('SPOD.sort',0)
	_dsort(&L[0,0],&P[0,0],&f[0],M,nBlks,nf)
	cr_stop('SPOD.sort',0)

	return L, P, f

@cr('SPOD.run')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def run(real[:,:] X, real[:] t, int nDFT=0, int nolap=0, remove_mean=True, W=None):
	r'''
	Run SPOD analysis of a matrix X.

	Args:
		X (np.ndarray): data matrix.
		t (np.ndarray): times at which the snapshots of X were collected
		nDFT (int, optional): number of points in each window (0 will set default value: ~10% nt)
		nolap (int, optional): number of overlap points between windows (0 will set default value: 50% nwin)
		remove_mean (bool, optional): whether or not to remove the mean flow (default, ``True``)
		W (np.ndarray, optional): 1D array with the diagonal of the inner-product weight matrix

	Returns:
		[(np.ndarray), (np.ndarray), (np.ndarray)]: where the first array is L, the modal energy spectra, the second array is  P, SPOD modes, whose spatial dimensions are identical to those of X and finally f is the frequency vectors
	''' 
	if real is double:
		W = None if W is None else np.ascontiguousarray(W,dtype=np.double)
		return _drun(X,t,nDFT,nolap,remove_mean,W)
	else:
		W = None if W is None else np.ascontiguousarray(W,dtype=np.float32)
		return _srun(X,t,nDFT,nolap,remove_mean,W)