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
	cdef int ii, jj, index
	cdef float tmp
	cdef float *v
	cdef int *idx
	v   = <float*>malloc(nf*sizeof(float))
	idx = <int*>malloc(nf*sizeof(int))
	# Copy to v
	for ii in range(nf):
		v[ii] = L[nBlks*ii + 0]
	# Obtain index array that sorts v
	c_ssort(v,idx,nf)
	free(v)
	# Reorder L, P and f arrays
	for ii in range(nf):
		index = nf-idx[ii]-1 # Reversed order
		# Swap f
		tmp      = f[index]
		f[index] = f[ii]
		f[ii]    = tmp
		# Swap L
		for jj in range(nBlks):
			tmp                 = L[nBlks*index + jj]
			L[nBlks*index + jj] = L[nBlks*ii + jj]
			L[nBlks*ii + jj]    = tmp
		# Swap P
		for jj in range(M*nBlks):
			tmp              = P[nf*jj + index]
			P[nf*jj + index] = P[nf*jj + ii]
			P[nf*jj + ii]    = tmp			
	free(idx)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _dsort(double *L, double *P, double *f, int M, int nBlks, int nf):
	cdef int ii, jj, index
	cdef double tmp
	cdef double *v
	cdef int *idx
	v   = <double*>malloc(nf*sizeof(double))
	idx = <int*>malloc(nf*sizeof(int))
	# Copy to v
	for ii in range(nf):
		v[ii] = L[nBlks*ii + 0]
	# Obtain index array that sorts v
	c_dsort(v,idx,nf)
	free(v)
	# Reorder L, P and f arrays
	for ii in range(nf):
		index = nf-idx[ii]-1 # Reversed order
		# Swap f
		tmp      = f[index]
		f[index] = f[ii]
		f[ii]    = tmp
		# Swap L
		for jj in range(nBlks):
			tmp                 = L[nBlks*index + jj]
			L[nBlks*index + jj] = L[nBlks*ii + jj]
			L[nBlks*ii + jj]    = tmp
		# Swap P
		for jj in range(M*nBlks):
			tmp              = P[nf*jj + index]
			P[nf*jj + index] = P[nf*jj + ii]
			P[nf*jj + ii]    = tmp			
	free(idx)


## SPOD run method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _srun(float[:,:] X, float[:] t, int nDFT, int nolap, int remove_mean):
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

	# Set frequency axis
	nf = <int>(ceil(nDFT/2)) + 1

	f = np.zeros((nf,)       ,dtype=np.float)
	L = np.zeros((nf,nBlks)  ,dtype=np.float)
	P = np.zeros((M*nBlks,nf),dtype=np.float)

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
	S  = <float*>malloc(M*nBlks*sizeof(float))
	V  = <np.complex64_t*>malloc(M*nBlks*sizeof(np.complex64_t))

	cr_start('SPOD.SVD',0)
	for ifreq in range(nf):
		# Load block in qf
		for i in range(M):
			for iblk in range(nBlks):
				winWeight = sqrt(nBlks) # reused variable
				qf[nBlks*i + iblk] = crealf(Q[nf*nBlks*i + nBlks*ifreq + iblk])/winWeight + cimagf(Q[nf*nBlks*i + nBlks*ifreq + iblk])/winWeight
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

	# Sort
	cr_start('SPOD.sort',0)
	_ssort(&L[0,0],&P[0,0],&f[0],M,nBlks,nf)
	cr_stop('SPOD.sort',0)

	return L, P, f

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _drun(double[:,:] X, double[:] t, int nDFT, int nolap, int remove_mean):
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
	S  = <double*>malloc(M*nBlks*sizeof(double))
	V  = <np.complex128_t*>malloc(M*nBlks*sizeof(np.complex128_t))

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
def run(real[:,:] X, real[:] t, int nDFT=0, int nolap=0, int remove_mean=True):
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
	if real is double:
		return _drun(X,t,nDFT,nolap,remove_mean)
	else:
		return _srun(X,t,nDFT,nolap,remove_mean)