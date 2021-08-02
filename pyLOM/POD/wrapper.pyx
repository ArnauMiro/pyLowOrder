#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for POD.
#
# Last rev: 09/07/2021
from __future__ import print_function, division

cimport cython
cimport numpy as np

import numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

from ..utils.cr     import cr_start, cr_stop
from ..utils.errors import raiseError


## Expose POD C functions
cdef extern from "pod.h":
	cdef void compute_temporal_mean(double *out, double *X, const int m, const int n)
	cdef void subtract_temporal_mean(double *out, double *X, double *X_mean, const int m, const int n)
	cdef void single_value_decomposition(double *U, double *S, double *V, double *Y, const int m, const int n)
	cdef  int compute_truncation_residual(double *S, double res, int n)
	cdef void compute_svd_truncation(double *U, double *S, double *VT, const int m, const int n, const int N)
	cdef void compute_power_spectral_density(double *PSD, double *y, const int n)
	cdef void compute_power_spectral_density_on_mode(double *PSD, double *V, const int n, const int m, const int transposed)

cdef extern from "matrix.h":
	cdef void transpose(double *A, const int m, const int n, const int bsz)


cdef np.ndarray[np.double_t,ndim=1] copy2array1D(double *src, int m):
	'''
	Copy a C array into a numpy array 2D
	Free C pointer
	'''
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((m,),dtype=np.double)
	memcpy(&out[0],src,m*sizeof(double))
	free(src)
	return out

cdef np.ndarray[np.double_t,ndim=2] copy2array2D(double *src, int m, int n):
	'''
	Copy a C array into a numpy array 2D
	Free C pointer
	'''
	cdef np.ndarray[np.double_t,ndim=2] out = np.zeros((m,n),dtype=np.double)
	memcpy(&out[0,0],src,m*n*sizeof(double))
	free(src)
	return out


## Cython function wrappers - for verification purposes
def temporal_mean(double[:,:] X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cr_start('POD.temporal_mean',0)
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((m,),dtype=np.double)
	# Compute temporal mean
	compute_temporal_mean(&out[0],&X[0,0],m,n)
	# Return
	cr_stop('POD.temporal_mean',0)
	return out

def subtract_mean(double[:,:] X, double[:] X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cr_start('POD.subtract_mean',0)
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] out = np.zeros((m,n),dtype=np.double)
	# Compute substract temporal mean
	subtract_temporal_mean(&out[0,0],&X[0,0],&X_mean[0],m,n)
	# Return
	cr_stop('POD.subtract_mean',0)
	return out

def svd(double[:,:] Y,int transpose_v=True,int bsz=0):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cr_start('POD.svd',0)
	cdef int m = Y.shape[0], n = Y.shape[1], mn = min(m,n)
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Compute SVD
	single_value_decomposition(&U[0,0],&S[0],&V[0,0],&Y[0,0],m,n)
	# Transpose V
	if transpose_v: transpose(&V[0,0],n,mn,bsz)
	cr_stop('POD.svd',0)
	return U,S,V

def residual(double[:] S, double r=1e-8):
	'''
	TODO: Function documentation
	'''
	cdef int ires = 0, n = S.shape[0]
	cr_start('POD.residual',0)
	ires = compute_truncation_residual(&S[0],r,n)
	cr_stop('POD.residual',0)
	return ires

def power_spectral_density(double [:] y):
	'''
	Compute the PSD of a signal y.
	'''
	cr_start('POD.psd',0)
	cdef int n = y.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] PSD = np.zeros((n,) ,dtype=np.double)
	# Compute PSD
	compute_power_spectral_density(&PSD[0],&y[0],n)
	cr_stop('POD.psd',0)
	return PSD


## POD run method
def run(double[:,:] X,double r=1e-8,int bsz=-1):
	'''
	Run POD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,n_temp_snapshots]: data matrix
		- r:                               target residual (optional)
		- bsz:                             bandsize for transpose (optional)
				if bsz is negative, transpose on V will not be performed

	Returns:
		- U:  are the POD modes.
		- S:  are the singular values.
		- V:  are the right singular vectors.
	'''
	cr_start('POD.run',0)
	# Variables
	cdef int m = X.shape[0], n = X.shape[1], mn = min(m,n), N, mN
	cdef double *X_mean
	cdef double *Ut
	cdef double *St
	cdef double *Vt
	# Output arrays
	cdef np.ndarray[np.double_t,ndim=2] U
	cdef np.ndarray[np.double_t,ndim=1] S
	cdef np.ndarray[np.double_t,ndim=2] V
	# Allocate memory
	X_mean = <double*>malloc(m*sizeof(double))
	Y      = <double*>malloc(m*n*sizeof(double))
	# Compute temporal mean
	compute_temporal_mean(X_mean,&X[0,0],m,n)
	# Compute substract temporal mean
	subtract_temporal_mean(&X[0,0],&X[0,0],X_mean,m,n)
	free(X_mean)
	# Compute SVD
	Ut = <double*>malloc(m*mn*sizeof(double))
	St = <double*>malloc(mn*sizeof(double))
	Vt = <double*>malloc(n*mn*sizeof(double))
	single_value_decomposition(Ut,St,Vt,&X[0,0],m,n)
	if bsz > 0: transpose(Vt,n,mn,bsz)
	N  = compute_truncation_residual(St,r,n)
	mN = min(m,N)
	compute_svd_truncation(Ut,St,Vt,m,n,N)
	# Copy memory to output arrays
	U = copy2array2D(Ut,m,mN)
	S = copy2array1D(St,mN)
	V = copy2array2D(Vt,N,mN)
	# Return
	cr_stop('POD.run',0)
	return U,S,V


## POD power density method
def PSD(double[:,:] V, double dt, int m=1, int transposed=True):
	'''
	Compute the power spectrum density of the matrix V
	and a given mode.

	Inputs:
		- V:          right singular vectors.
		- dt:         timestep.
		- m:          mode to perform PSD starting at 1.
		- transposed: whether we input V or Vt.

	Outputs:
		- PSD:  Power spectrum density.
		- freq: Associated frequencies.
	'''
	cr_start('POD.PSD',0)
	cdef int ii, n = V.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] PSD  = np.zeros((n,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] freq = np.zeros((n,) ,dtype=np.double)
	# Compute PSD of mode m
	compute_power_spectral_density_on_mode(&PSD[0],&V[0,0],n,m-1,transposed)
	# Compute frequency array (in Hz)
	for ii in range(n):
		freq[ii] = 1./dt/n*ii
	# Return
	cr_stop('POD.PSD',0)
	return PSD, freq


## POD reconstruct method
