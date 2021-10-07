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
#		memcpy(Y_copy,&Y[0,n1],m*n*sizeof(double))
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