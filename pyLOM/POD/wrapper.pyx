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
	cdef void   compute_temporal_mean(double *out, double *X, const int m, const int n)
	cdef void   subtract_temporal_mean(double *out, double *X, double *X_mean, const int m, const int n)
	cdef void   single_value_decomposition(double *U, double *S, double *V, double *Y, const int m, const int n)
	cdef  int   compute_truncation_residual(double *S, double res, int n)
	cdef void   compute_svd_truncation(double *Ur, double *Sr, double *VTr, double *U, double *S, double *VT, const int m, const int n, const int N)
	cdef void   TSQR_single_value_decomposition(double *Ui, double *S, double *VT, double *Ai, const int m, const int n)
	cdef void   compute_power_spectral_density(double *PSD, double *y, const int n)
	cdef void   compute_power_spectral_density_on_mode(double *PSD, double *V, const int n, const int m, const int transposed)
	cdef void   compute_reconstruct_svd(double *X, double *Ur, double *Sr, double *VTr, const int m, const int n, const int N)
	cdef double compute_RMSE(double *X_POD, double *X, const int m, const int n)

cdef extern from "matrix.h":
	cdef void transpose(double *A, const int m, const int n, const int bsz)
	cdef double compute_norm(double *A, int start, int n)


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

def svd(double[:,:] Y,int do_copy=True,int bsz=-1):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cr_start('POD.svd',0)
	cdef int m = Y.shape[0], n = Y.shape[1], mn = min(m,n)
	cdef double *Y_copy
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Compute SVD
	if do_copy:
		Y_copy = <double*>malloc(m*n*sizeof(double))
		memcpy(Y_copy,&Y[0,0],m*n*sizeof(double))
		single_value_decomposition(&U[0,0],&S[0],&V[0,0],Y_copy,m,n)
		free(Y_copy)
	else:
		single_value_decomposition(&U[0,0],&S[0],&V[0,0],&Y[0,0],m,n)
	# Transpose V
	if bsz >= 0: transpose(&V[0,0],n,mn,bsz)
	cr_stop('POD.svd',0)
	return U,S,V

def tsqr_svd(double[:,:] Y,int bsz=-1):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cr_start('POD.tsqr_svd',0)
	cdef int m = Y.shape[0], n = Y.shape[1], mn = min(m,n)
	cdef double *Y_copy
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Compute SVD using TSQR algorithm
	TSQR_single_value_decomposition(&U[0,0],&S[0],&V[0,0],&Y[0,0],m,n)
	# Transpose V
	if bsz >= 0: transpose(&V[0,0],n,mn,bsz)
	cr_stop('POD.tsqr_svd',0)
	return U,S,V

def residual(double[:] S, double r=1e-8):
	'''
	Computes the residual and the point where to
	truncate the matrices specifying a target residual.
	'''
	cdef int ires = 0, n = S.shape[0]
	cdef double outres = 0.
	cr_start('POD.residual',0)
	ires   = compute_truncation_residual(&S[0],r,n)
	outres = compute_norm(&S[0],ires,n) #TODO: fix not working (see example_POD_matlab)
	cr_stop('POD.residual',0)
	return ires, outres

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

def RMSE(double[:,:] X_POD, double[:,:] X):
	'''
	Compute RMSE between X_POD and X
	'''
	cr_start('POD.RMSE',0)
	cdef int m = X.shape[0], n = X.shape[1]
	cdef double rmse = 0.
	rmse = compute_RMSE(&X_POD[0,0],&X[0,0],m,n)
	cr_stop('POD.RMSE',0)
	return rmse


## POD run method
def run(double[:,:] X,int remove_mean=True, int bsz=-1):
	'''
	Run POD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,n_temp_snapshots]: data matrix
                - remove_mean:                     whether or not to remove the mean flow
		- bsz:                             bandsize for transpose (optional)
				if bsz is negative, transpose on V will not be performed

	Returns:
		- U:  are the POD modes.
		- S:  are the singular values.
		- V:  are the right singular vectors.
	'''
	cr_start('POD.run',0)
	# Variables
	cdef int m = X.shape[0], n = X.shape[1], mn = min(m,n)
	cdef double *X_mean
	cdef double *Y
	# Output arrays
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Allocate memory
	Y      = <double*>malloc(m*n*sizeof(double))
	if remove_mean:
		X_mean = <double*>malloc(m*sizeof(double))
		# Compute temporal mean
		compute_temporal_mean(X_mean,&X[0,0],m,n)
		# Compute substract temporal mean
		subtract_temporal_mean(Y,&X[0,0],X_mean,m,n)
		free(X_mean)
	else:
		memcpy(Y,&X[0,0],m*n*sizeof(double))
	# Compute SVD
	TSQR_single_value_decomposition(&U[0,0],&S[0],&V[0,0],Y,m,n)
	free(Y)
	if bsz >= 0: transpose(&V[0,0],n,mn,bsz)
	# Return
	cr_stop('POD.run',0)
	return U,S,V


## POD truncate method
def truncate(double[:,:] U, double[:] S, double[:,:] V, double r=1e-8):
	'''
	Truncate POD matrices (U,S,V) given a residual r.

	Inputs:
		- U(m,n)  are the POD modes.
		- S(n)    are the singular values.
		- V(n,n)  are the right singular vectors.
		- r       target residual (default 1e-8)

	Returns:
		- U(m,N)  are the POD modes (truncated at N).
		- S(N)    are the singular values (truncated at N).
		- V(n,N)  are the right singular vectors (truncated at N).
	'''
	cr_start('POD.truncate',0)
	cdef int m = U.shape[0], n = S.shape[0], mn = min(m,n), N, mN
	# Output arrays
	cdef np.ndarray[np.double_t,ndim=2] Ur = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] Sr = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Vr = np.zeros((n,mn),dtype=np.double)
	# Compute N using S
	N  = compute_truncation_residual(&S[0],r,n)
	mN = min(m,N)
	# Allocate output arrays
	Ur = np.zeros((m,N),dtype=np.double)
	Sr = np.zeros((N,) ,dtype=np.double)
	Vr = np.zeros((N,n),dtype=np.double)
	# Truncate
	compute_svd_truncation(&Ur[0,0],&Sr[0],&Vr[0,0],&U[0,0],&S[0],&V[0,0],m,n,N)
	# Return
	cr_stop('POD.truncate',0)
	return Ur, Sr, Vr


## POD power density method
def PSD(double[:,:] V, double dt, int m=1, int transposed=True):
	'''
	Compute the power spectrum density of the matrix V
	and a given mode.

	Inputs:
		- V:          right singular vectors.
		- dt:         timestep.
		- m:          mode to perform PSD starting at 1.
		- transposed: whether we input V or VT (default true->VT).

	Outputs:
		- PSD:  Power spectrum density.
		- freq: Associated frequencies.
	'''
	cr_start('POD.PSD',0)
	cdef int ii, n = V.shape[1] if transposed else V.shape[0]
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
def reconstruct(double[:,:] U, double[:] S, double[:,:] V, overwrite=False):
	'''
	Reconstruct the flow given the POD decomposition matrices
	that can be possibly truncated.

	Inputs:
		- U(m,N)  are the POD modes.
		- S(N)    are the singular values.
		- V(N,n)  are the right singular vectors.

	Outputs
		- X(m,n)  is the reconstructed flow.
	'''
	cr_start('POD.reconstruct',0)
	cdef int m = U.shape[0], N = S.shape[0], n = V.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] X = np.zeros((m,n),dtype=np.double)
	cdef double *VT
	# Call C function
	if not overwrite:
		VT = <double*>malloc(N*n*sizeof(double))
		memcpy(VT,&V[0,0],N*n*sizeof(double))
		compute_reconstruct_svd(&X[0,0],&U[0,0],&S[0],VT,m,n,N)
		free(VT)
	else:
		compute_reconstruct_svd(&X[0,0],&U[0,0],&S[0],&V[0,0],m,n,N)
	# Return
	cr_stop('POD.reconstruct',0)
	return X
