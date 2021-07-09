#!/usr/bin/env cpython
#
# pyLOM - Pythn Low Order Modeling.
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

from .utils.cr     import cr_start, cr_stop
from .utils.errors import raiseError


## Expose POD C functions
cdef extern from "pod.h":
	cdef void compute_temporal_mean(double *out, double *X, const int m, const int n)
	cdef void subtract_temporal_mean(double *out, double *X, double *X_mean, const int m, const int n)
	cdef void single_value_decomposition(double *U, double *S, double *V, double *Y, const int m, const int n)


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
	# Compute temporal mean
	subtract_temporal_mean(&out[0,0],&X[0,0],&X_mean[0],m,n)
	# Return
	cr_stop('POD.subtract_mean',0)
	return out	

def svd(double[:,:] Y):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cr_start('POD.svd',0)
	cdef int m = Y.shape[0], n = Y.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((n,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,n),dtype=np.double)
	# Compute SVD
	single_value_decomposition(&U[0,0],&S[0],&V[0,0],&Y[0,0],m,n)
	cr_stop('POD.svd',0)
	return U,S,V


## POD run method
def run():
	'''
	Run POD
	'''
	cr_start('POD.run',0)
	cr_stop('POD.run',0)