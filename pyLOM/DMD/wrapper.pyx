#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for DMD.
#
# Last rev: 30/09/2021
from __future__ import print_function, division

cimport cython
cimport numpy as np

import numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

from ..utils.cr     import cr_start, cr_stop
from ..utils.errors import raiseError


## Expose DMD C functions
cdef extern from "pod.h":
	cdef void   compute_temporal_mean(double *out, double *X, const int m, const int n)
	cdef void   subtract_temporal_mean(double *out, double *X, double *X_mean, const int m, const int n)
	cdef void   single_value_decomposition(double *U, double *S, double *V, double *Y, const int m, const int n)
	cdef  int   compute_truncation_residual(double *S, double res, int n)
	cdef void   compute_svd_truncation(double *Ur, double *Sr, double *VTr, double *U, double *S, double *VT, const int m, const int n, const int N)
	cdef void   compute_reconstruct_svd(double *X, double *Ur, double *Sr, double *VTr, const int m, const int n, const int N)
	cdef double compute_RMSE(double *X_POD, double *X, const int m, const int n)

cdef extern from "dmd.h":
	cdef void compute_eigen(double *delta, double *w, double *veps, double *A, const int m, const int n)

cdef extern from "matrix.h":
	cdef void transpose(double *A, const int m, const int n, const int bsz)
	cdef double compute_norm(double *A, int start, int n)


def svd(double[:,:] Y,int n1,int n2,int do_copy=True,int bsz=-1):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cr_start('POD.svd',0)
	cdef int ii, jj, m = Y.shape[0], n = n2-n1, mn = min(m,n)
	cdef double *Y_copy
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Compute SVD
	if do_copy:
		Y_copy = <double*>malloc(m*n*sizeof(double))
		for ii in range(m):
			for jj in range(n1,n2):
				Y_copy[n*ii+jj] = Y[ii,jj]
		single_value_decomposition(&U[0,0],&S[0],&V[0,0],Y_copy,m,n)
		free(Y_copy)
	else:
		single_value_decomposition(&U[0,0],&S[0],&V[0,0],&Y[0,0],m,n)
	# Transpose V
	if bsz >= 0: transpose(&V[0,0],n,mn,bsz)
	cr_stop('POD.svd',0)
	return U,S,V

def eigen(double[:,:] Y):
	'''
	Eigenvalues and eigenvectors using Lapack.
		delta(n)  are the real eigenvalues.
		w(n)      are the imaginary eigenvalues.
		v(n,n)    are the right eigenvectors.
	'''
	cr_start('DMD.eigen',0)
	cdef int m = Y.shape[0], n = Y.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] delta = np.zeros((n,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] w     = np.zeros((n,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] v     = np.zeros((n,n),dtype=np.double)
	# Compute eigenvalues and eigenvectors
	compute_eigen(&delta[0],&w[0],&v[0,0],&Y[0,0],m,n)
	cr_stop('DMD.eigen',0)
	return delta,w,v

def matrix_split(X):
	'''
	Splits a matrix into two:
		X1 Excluding the last snapshot
		X2 Excluding the first snapshot
	'''
	return X[:, :-1], X[:, 1:]

def project_POD_basis(U, X, V, S):
	'''
	Projects matrix A (Jacobian of the snapshots) to the POD basis
	'''
	return np.matmul(np.matmul(np.matmul(np.transpose(U), X), np.transpose(V)), np.diag(1/S))

def build_complex_eigenvectors(w, eigImag):
	wComplex = np.zeros(w.shape, dtype = 'complex_')
	ivec = 0
	while ivec < w.shape[1] - 1:
		if eigImag[ivec] > np.finfo(np.double).eps:
			wComplex[:, ivec]     = w[:, ivec] + w[:, ivec + 1]*1j
			wComplex[:, ivec + 1] = w[:, ivec] - w[:, ivec + 1]*1j
			ivec += 2
		else:
			wComplex[:, ivec] = w[:, ivec] + 0*1j
			ivec = ivec + 1
	return wComplex

def polar(real, imag):
	modulus = np.sqrt(real*real + imag*imag)
	arg     = np.arctan2(imag, real)
	return modulus, arg

def frequency_damping(eigReal, eigImag, dt):
	#Compute modulus and argument of the eigenvalues
	eigModulus, eigArg = polar(eigReal, eigImag)

	#Computation of the damping ratio of the mode
	delta = np.log(eigModulus)/dt

	#Computation of the frequency of the mode
	omega = eigArg/dt

	return delta, omega, eigModulus, eigArg

def mode_computation(X, V, S, W):
	return np.matmul(np.matmul(np.matmul(X, np.transpose(V)), np.diag(1/S)), W)

def vandermonde(eigReal, eigImag, shape0, shape1):
	eigModulus, eigArg = polar(eigReal, eigImag)
	Vand  = np.zeros((shape0, shape1), dtype = 'complex_')
	for icol in range(shape1):
		VandModulus   = eigModulus**icol
		VandArg       = eigArg*icol
		Vand[:, icol] = VandModulus*np.cos(VandArg) + VandModulus*np.sin(VandArg)*1j
	return Vand

def amplitude_jovanovic(eigReal, eigImag, shape0, shape1, wComplex, S, V):
	Vand = vandermonde(eigReal, eigImag, shape0, shape1)
	P    = np.matmul(np.transpose(np.conj(wComplex)), wComplex)*np.conj(np.matmul(Vand, np.transpose(np.conj(Vand))))
	Pl   = np.linalg.cholesky(P)
	G    = np.matmul(np.diag(S), V)
	q    = np.conj(np.diag(np.matmul(np.matmul(Vand, np.transpose(np.conj(G))), wComplex)))
	bJov = np.matmul(np.linalg.inv(np.transpose(np.conj(Pl))), np.matmul(np.linalg.inv(Pl), q)) #Amplitudes according to Jovanovic 2014
	return bJov
