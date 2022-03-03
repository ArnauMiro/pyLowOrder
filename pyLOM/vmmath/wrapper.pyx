#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

cimport cython
cimport numpy as np

import numpy as np
from mpi4py  import MPI

from libc.stdlib   cimport malloc, free
from libc.string   cimport memcpy, memset
from mpi4py        cimport MPI
from mpi4py.libmpi cimport MPI_Comm

from ..utils.cr     import cr_start, cr_stop
from ..utils.errors import raiseError


## Expose C functions
cdef extern from "vector_matrix.h":
	cdef void   c_transpose   "transpose"(double *A, const int m, const int n)
	cdef double c_vector_norm "vector_norm"(double *v, int start, int n)
	cdef void   c_matmul      "matmul"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_vecmat      "vecmat"(double *v, double *A, const int m, const int n)
	cdef int    c_eigen       "eigen"(double *real, double *imag, double *vecs, double *A, const int m, const int n)
	cdef double c_RMSE        "RMSE"(double *A, double *B, const int m, const int n, MPI_Comm comm)
cdef extern from "averaging.h":
	cdef void c_temporal_mean "temporal_mean"(double *out, double *X, const int m, const int n)
	cdef void c_subtract_mean "subtract_mean"(double *out, double *X, double *X_mean, const int m, const int n)
cdef extern from "svd.h":
	cdef int c_qr       "qr"      (double *Q, double *R, double *A, const int m, const int n)
	cdef int c_svd      "svd"     (double *U, double *S, double *V, double *Y, const int m, const int n)
	cdef int c_tsqr     "tsqr"    (double *Qi, double *R, double *Ai, const int m, const int n, MPI_Comm comm)
	cdef int c_tsqr_svd "tsqr_svd"(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm)
cdef extern from "fft.h":
	cdef void c_fft "fft"(double *psd, double *y, const double dt, const int n)
	cdef void c_nfft "nfft"(double *psd, double *t, double* y, const int n)

## Cython functions
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def transpose(double[:,:] A):
	'''
	Transposed of matrix A
	'''
	cr_start('math.transpose',0)
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] At = np.zeros((m,n),dtype=np.double)
	memcpy(&At[0,0],&A[0,0],m*n*sizeof(double))
	c_transpose(&At[0,0],m,n)
	cr_stop('math.transpose',0)
	return At

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vector_norm(double[:] v, int start=0):
	'''
	L2 norm of a vector
	'''
	cr_start('math.vector_norm',0)
	cdef int n = v.shape[0]
	cdef double norm = 0.
	norm = c_vector_norm(&v[0],start,n)
	cr_stop('math.vector_norm',0)	
	return norm

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def matmul(double[:,:] A, double[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	cr_start('math.matmul',0)
	cdef int m = A.shape[0], k = A.shape[1], n = B.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] C = np.zeros((m,n),dtype=np.double)
	c_matmul(&C[0,0],&A[0,0],&B[0,0],m,n,k)
	cr_stop('math.matmul',0)	
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vecmat(double[:] v, double[:,:] A):
	'''
	Vector times a matrix C = v x A
	'''
	cr_start('math.vecmat',0)
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] C = np.zeros((m,n),dtype=np.double)
	memcpy(&C[0,0],&A[0,0],m*n*sizeof(double))
	c_vecmat(&v[0],&C[0,0],m,n)
	cr_stop('math.vecmat',0)	
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def eigen(double[:,:] A):
	'''
	Eigenvalues and eigenvectors using Lapack.
		real(n)   are the real eigenvalues.
		imag(n)   are the imaginary eigenvalues.
		vecs(n,n) are the right eigenvectors.
	'''
	cr_start('math.eigen',0)
	cdef int m = A.shape[0], n = A.shape[1], retval
	cdef np.ndarray[np.double_t,ndim=1] real = np.zeros((n,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] imag = np.zeros((n,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] vecs = np.zeros((n,n),dtype=np.double)
	# Compute eigenvalues and eigenvectors
	retval = c_eigen(&real[0],&imag[0],&vecs[0,0],&A[0,0],m,n)
	cr_stop('math.eigen',0)
	if not retval == 0: raiseError('Problems computing SVD!')
	return real,imag,vecs

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def temporal_mean(double[:,:] X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cr_start('math.temporal_mean',0)
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((m,),dtype=np.double)
	# Compute temporal mean
	c_temporal_mean(&out[0],&X[0,0],m,n)
	# Return
	cr_stop('math.temporal_mean',0)
	return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def subtract_mean(double[:,:] X, double[:] X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cr_start('math.subtract_mean',0)
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] out = np.zeros((m,n),dtype=np.double)
	# Compute substract temporal mean
	c_subtract_mean(&out[0,0],&X[0,0],&X_mean[0],m,n)
	# Return
	cr_stop('math.subtract_mean',0)
	return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def qr(double[:,:] A):
	'''
	QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cr_start('math.qr')
	cdef int retval, m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] Q = np.zeros((m,n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] R = np.zeros((n,n),dtype=np.double)
	retval = c_qr(&Q[0,0],&R[0,0],&A[0,0],m,n)
	cr_stop('math.qr')
	if not retval == 0: raiseError('Problems computing QR factorization!')
	return Q,R

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def svd(double[:,:] A, int do_copy=True):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cr_start('math.svd',0)
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef double *Y_copy
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Compute SVD
	if do_copy:
		Y_copy = <double*>malloc(m*n*sizeof(double))
		memcpy(Y_copy,&A[0,0],m*n*sizeof(double))
		retval = c_svd(&U[0,0],&S[0],&V[0,0],Y_copy,m,n)
		free(Y_copy)
	else:
		retval = c_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n)
	cr_stop('math.svd',0)
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def tsqr(double[:,:] A):
	'''
	Parallel QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cr_start('math.tsqr',0)
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.double_t,ndim=2] Qi = np.zeros((m,n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] R  = np.zeros((n,n),dtype=np.double)
	# Compute SVD using TSQR algorithm
	retval = c_tsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	cr_stop('math.tsqr',0)
	if not retval == 0: raiseError('Problems computing TSQR!')
	return Qi,R

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def tsqr_svd(double[:,:] A):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cr_start('math.tsqr_svd',0)
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Compute SVD using TSQR algorithm
	retval = c_tsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	cr_stop('math.tsqr_svd',0)
	if not retval == 0: raiseError('Problems computing TSQR SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def fft(double [:] t, double[:] y, int equispaced=True):
	'''
	Compute the fft of a signal y that is sampled at a
	constant timestep. Return the frequency and PSD
	'''
	cr_start('math.fft',0)
	cdef int n = y.shape[0]
	cdef double ts = t[1] - t[0]
	cdef np.ndarray[np.double_t,ndim=1] f   = np.zeros((n,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] PSD = np.zeros((n,) ,dtype=np.double)
	memcpy(&f[0],&y[0],n*sizeof(double))
	if equispaced:
		c_fft(&PSD[0],&f[0],ts,n)
	else:
		c_nfft(&PSD[0],&f[0],&t[0],n)
	cr_stop('math.fft',0)
	return f, PSD

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def RMSE(double[:,:] A, double[:,:] B):
	'''
	Compute RMSE between X_POD and X
	'''
	cr_start('math.RMSE',0)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef int m = A.shape[0], n = B.shape[1]
	cdef double rmse = 0.
	rmse = c_RMSE(&A[0,0],&B[0,0],m,n,MPI_COMM.ob_mpi)
	cr_stop('math.RMSE',0)
	return rmse