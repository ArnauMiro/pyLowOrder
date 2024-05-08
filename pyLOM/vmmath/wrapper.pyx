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
from libc.math     cimport sqrt, atan2
from mpi4py        cimport MPI
from mpi4py.libmpi cimport MPI_Comm

from ..utils.cr     import cr
from ..utils.errors import raiseError


## Expose C functions
cdef extern from "vector_matrix.h" nogil:
	# Double precision
	cdef void   c_transpose        "transpose"(double *A, double *B, const int m, const int n)
	cdef double c_vector_norm      "vector_norm"(double *v, int start, int n)
	cdef void   c_matmult          "matmult"(double *C, double *A, double *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_matmul           "matmul"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_matmulp          "matmulp"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_vecmat           "vecmat"(double *v, double *A, const int m, const int n)
	cdef int    c_inverse          "inverse"(double *A, int N, char *UoL)
	cdef double c_RMSE             "RMSE"(double *A, double *B, const int m, const int n, MPI_Comm comm)
	cdef void   c_sort             "sort"(double *v, int *index, int n)
	# Double complex precision
	cdef void   c_zmatmult         "zmatmult"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_zmatmul          "zmatmul"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k)
	cdef void 	c_zmatmulp         "zmatmulp"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k)
	cdef void   c_zvecmat          "zvecmat"(np.complex128_t *v, np.complex128_t *A, const int m, const int n)
	cdef int    c_zinverse         "zinverse"(np.complex128_t *A, int N, char *UoL)
	cdef int    c_eigen            "eigen"(double *real, double *imag, np.complex128_t *vecs, double *A, const int m, const int n)
	cdef int    c_cholesky         "cholesky"(np.complex128_t *A, int N)
	cdef void   c_vandermonde      "vandermonde"(np.complex128_t *Vand, double *real, double *imag, int m, int n)
	cdef void   c_vandermonde_time "vandermondeTime"(np.complex128_t *Vand, double *real, double *imag, int m, int n, double* t)
	cdef void   c_zsort            "zsort"(np.complex128_t *v, int *index, int n)
cdef extern from "averaging.h":
	cdef void c_temporal_mean "temporal_mean"(double *out, double *X, const int m, const int n)
	cdef void c_subtract_mean "subtract_mean"(double *out, double *X, double *X_mean, const int m, const int n)
cdef extern from "svd.h":
	# Double precision
	cdef int c_qr        "qr"      (double *Q, double *R, double *A, const int m, const int n)
	cdef int c_svd       "svd"     (double *U, double *S, double *V, double *Y, const int m, const int n)
	cdef int c_tsqr      "tsqr"    (double *Qi, double *R, double *Ai, const int m, const int n, MPI_Comm comm)
	cdef int c_tsqr_svd  "tsqr_svd"(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm)
	# Double complex precision
	cdef int c_zqr       "zqr"      (np.complex128_t *Q, np.complex128_t *R, np.complex128_t *A, const int m, const int n)
	cdef int c_zsvd      "zsvd"     (np.complex128_t *U, np.double_t *S, np.complex128_t *V, np.complex128_t *Y, const int m, const int n)
	cdef int c_ztsqr     "ztsqr"    (np.complex128_t *Qi, np.complex128_t *R, np.complex128_t *Ai, const int m, const int n, MPI_Comm comm)
	cdef int c_ztsqr_svd "ztsqr_svd"(np.complex128_t *Ui, np.double_t *S, np.complex128_t *VT, np.complex128_t *Ai, const int m, const int n, MPI_Comm comm)
cdef extern from "fft.h":
	cdef int USE_FFTW3 "_USE_FFTW3"
	cdef void c_fft "fft"(double *psd, double *y, const double dt, const int n)
	cdef void c_nfft "nfft"(double *psd, double *t, double* y, const int n)


## Fused type between double and complex
ctypedef fused double_complex:
	double
	np.complex128_t


## Cython functions
@cr('math.transpose')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def transpose(double[:,:] A):
	'''
	Transposed of matrix A
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] At = np.zeros((n,m),dtype=np.double)
	c_transpose(&A[0,0], &At[0,0], m,n)
	return At

@cr('math.vector_norm')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vector_norm(double[:] v, int start=0):
	'''
	L2 norm of a vector
	'''
	cdef int n = v.shape[0]
	cdef double norm = 0.
	norm = c_vector_norm(&v[0],start,n)
	return norm

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dmatmul(double[:,:] A, double[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	cdef int m = A.shape[0], k = A.shape[1], n = B.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] C = np.zeros((m,n),dtype=np.double)
	c_matmul(&C[0,0],&A[0,0],&B[0,0],m,n,k)
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zmatmul(np.complex128_t[:,:] A, np.complex128_t[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	cdef int m = A.shape[0], k = A.shape[1], n = B.shape[1]
	cdef np.ndarray[np.complex128_t,ndim=2] C = np.zeros((m,n),dtype=np.complex128)
	c_zmatmul(&C[0,0],&A[0,0],&B[0,0],m,n,k)
	return C

@cr('math.matmul')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def matmul(double_complex[:,:] A, double_complex[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	if double_complex is np.complex128_t:
		return _zmatmul(A,B)
	else:
		return _dmatmul(A,B)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dmatmulp(double[:,:] A, double[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	cdef int m = A.shape[0], k = A.shape[1], n = B.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] C = np.zeros((m,n),dtype=np.double)
	c_matmulp(&C[0,0],&A[0,0],&B[0,0],m,n,k)
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zmatmulp(np.complex128_t[:,:] A, np.complex128_t[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	cdef int m = A.shape[0], k = A.shape[1], n = B.shape[1]
	cdef np.ndarray[np.complex128_t,ndim=2] C = np.zeros((m,n),dtype=np.complex128)
	c_zmatmulp(&C[0,0],&A[0,0],&B[0,0],m,n,k)
	return C

@cr('math.matmulp')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def matmulp(double_complex[:,:] A, double_complex[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	if double_complex is np.complex128_t:
		return _zmatmulp(A,B)
	else:
		return _dmatmulp(A,B)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dvecmat(double[:] v, double[:,:] A):
	'''
	Vector times a matrix C = v x A
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] C = np.zeros((m,n),dtype=np.double)
	memcpy(&C[0,0],&A[0,0],m*n*sizeof(double))
	c_vecmat(&v[0],&C[0,0],m,n)
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zvecmat(np.complex128_t[:] v, np.complex128_t[:,:] A):
	'''
	Vector times a matrix C = v x A
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.complex128_t,ndim=2] C = np.zeros((m,n),dtype=np.complex128)
	memcpy(&C[0,0],&A[0,0],m*n*sizeof(np.complex128_t))
	c_zvecmat(&v[0],&C[0,0],m,n)
	return C

@cr('math.vecmat')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vecmat(double_complex[:] v, double_complex[:,:] A):
	'''
	Vector times a matrix C = v x A
	'''
	if double_complex is np.complex128_t:
		return _zvecmat(v,A)
	else:
		return _dvecmat(v,A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _dsort(double[:] v, int[:] index):
	'''
	Sort a vector
	'''
	cdef int n = v.shape[0]
	c_sort(&v[0],&index[0],n)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _zsort(np.complex128_t[:] v, int[:] index):
	'''
	Sort a vector
	'''
	cdef int n = v.shape[0]
	c_zsort(&v[0],&index[0],n)

@cr('math.argsort')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def argsort(double_complex[:] v):
	'''
	Returns the indices that sort a vector
	'''
	cdef int n = v.shape[0]
	cdef np.ndarray[np.int32_t,ndim=1] index = np.zeros((n,),dtype=np.int32)
	if double_complex is np.complex128_t:
		return _zsort(v,index)
	else:
		return _dsort(v,index)

@cr('math.eigen')
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
	cdef int m = A.shape[0], n = A.shape[1], retval
	cdef np.ndarray[np.double_t,ndim=1] real = np.zeros((n,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] imag = np.zeros((n,),dtype=np.double)
	cdef np.ndarray[np.complex128_t,ndim=2] vecs = np.zeros((n,n),dtype=np.complex128)
	# Compute eigenvalues and eigenvectors
	retval = c_eigen(&real[0],&imag[0],&vecs[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing eigenvalues!')
	return real,imag,vecs

@cr('math.temporal_mean')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def temporal_mean(double[:,:] X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((m,),dtype=np.double)
	# Compute temporal mean
	c_temporal_mean(&out[0],&X[0,0],m,n)
	# Return
	return out

@cr('math.polar')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def polar(double[:] real, double[:] imag):
	'''
	Present a complex number in its polar form given its real and imaginary part
	'''
	cdef int i, n = real.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] mod = np.zeros((n,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] arg = np.zeros((n,),dtype=np.double)
	for i in range(n):
		mod[i] = sqrt(real[i]*real[i] + imag[i]*imag[i])
		arg[i] = atan2(imag[i], real[i])
	return mod, arg

@cr('math.subtract_mean')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def subtract_mean(double[:,:] X, double[:] X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] out = np.zeros((m,n),dtype=np.double)
	# Compute substract temporal mean
	c_subtract_mean(&out[0,0],&X[0,0],&X_mean[0],m,n)
	# Return
	return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dqr(double[:,:] A):
	'''
	QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cdef int retval, m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] Q = np.zeros((m,n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] R = np.zeros((n,n),dtype=np.double)
	retval = c_qr(&Q[0,0],&R[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing QR factorization!')
	return Q,R

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _zqr(np.complex128_t[:,:] A):
	'''
	QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cdef int retval, m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.complex128_t,ndim=2] Q = np.zeros((m,n),dtype=np.complex128)
	cdef np.ndarray[np.complex128_t,ndim=2] R = np.zeros((n,n),dtype=np.complex128)
	retval = c_zqr(&Q[0,0],&R[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing QR factorization!')
	return Q,R

@cr('math.qr')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def qr(double_complex[:,:] A):
	'''
	QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	if double_complex is np.complex128_t:
		return _zqr(A)
	else:
		return _dqr(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dsvd(double[:,:] A, int do_copy=True):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
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
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _zsvd(np.complex128_t[:,:] A, int do_copy=True):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef np.complex128_t *Y_copy
	cdef np.ndarray[np.complex128_t,ndim=2] U = np.zeros((m,mn),dtype=np.complex128)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.complex128_t,ndim=2] V = np.zeros((n,mn),dtype=np.complex128)
	# Compute SVD
	if do_copy:
		Y_copy = <np.complex128_t*>malloc(m*n*sizeof(np.complex128_t))
		memcpy(Y_copy,&A[0,0],m*n*sizeof(np.complex128_t))
		retval = c_zsvd(&U[0,0],&S[0],&V[0,0],Y_copy,m,n)
		free(Y_copy)
	else:
		retval = c_zsvd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cr('math.svd')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def svd(double_complex[:,:] A, int do_copy=True):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	if double_complex is np.complex128_t:
		return _zsvd(A,do_copy)
	else:
		return _dsvd(A,do_copy)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dtsqr(double[:,:] A):
	'''
	Parallel QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.double_t,ndim=2] Qi = np.zeros((m,n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] R  = np.zeros((n,n),dtype=np.double)
	# Compute SVD using TSQR algorithm
	retval = c_tsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing TSQR!')
	return Qi,R

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _ztsqr(np.complex128_t[:,:] A):
	'''
	Parallel QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.complex128_t,ndim=2] Qi = np.zeros((m,n),dtype=np.complex128)
	cdef np.ndarray[np.complex128_t,ndim=2] R  = np.zeros((n,n),dtype=np.complex128)
	# Compute SVD using TSQR algorithm
	retval = c_ztsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing TSQR!')
	return Qi,R

@cr('math.tsqr')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def tsqr(double_complex[:,:] A):
	'''
	Parallel QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	if double_complex is np.complex128_t:
		return _ztsqr(A)
	else:
		return _dtsqr(A)

### FIX

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dtsqr_svd(double[:,:] A):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Compute SVD using TSQR algorithm
	retval = c_tsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing TSQR SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _ztsqr_svd(np.complex128_t[:,:] A):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.complex128_t,ndim=2] U = np.zeros((m,mn),dtype=np.complex128)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.complex128_t,ndim=2] V = np.zeros((n,mn),dtype=np.complex128)
	# Compute SVD using TSQR algorithm
	retval = c_ztsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing TSQR SVD!')
	return U,S,V

@cr('math.tsqr_svd')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def tsqr_svd(double_complex[:,:] A):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	if double_complex is np.complex128_t:
		return _ztsqr_svd(A)
	else:
		return _dtsqr_svd(A)

@cr('math.fft')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def fft(double [:] t, double[:] y, int equispaced=True):
	'''
	Compute the fft of a signal y that is sampled at a
	constant timestep. Return the frequency and PSD
	'''
	cdef int n = y.shape[0]
	cdef double ts = t[1] - t[0], k_left
	cdef np.ndarray[np.double_t,ndim=1]     x
	cdef np.ndarray[np.complex128_t,ndim=1] yf
	cdef np.ndarray[np.double_t,ndim=1] f   = np.zeros((n,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] PSD = np.zeros((n,) ,dtype=np.double)
	memcpy(&f[0],&y[0],n*sizeof(double))
	if equispaced:
		c_fft(&PSD[0],&f[0],ts,n)
	else:
		if USE_FFTW3:
			c_nfft(&PSD[0],&f[0],&t[0],n)
		else:
			import nfft
			# Compute sampling frequency
			k_left = (t.shape[0]-1.)/2.
			f      = (np.arange(t.shape[0],dtype=np.double)-k_left)/t[n-1]
			# Compute power spectra using fft
			x   = -0.5 + np.arange(t.shape[0],dtype=np.double)/t.shape[0]
			yf  = nfft.nfft_adjoint(x,y,len(t))
			PSD = np.real(yf*np.conj(yf))/y.shape[0]
	return f, PSD

@cr('math.RMSE')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def RMSE(double[:,:] A, double[:,:] B):
	'''
	Compute RMSE between X_POD and X
	'''
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef int m = A.shape[0], n = B.shape[1]
	cdef double rmse = 0.
	rmse = c_RMSE(&A[0,0],&B[0,0],m,n,MPI_COMM.ob_mpi)
	return rmse

@cr('math.cholesky')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cholesky(np.complex128_t[:,:] A):
	'''
	Compute the Lower Cholesky decomposition of matrix A. The C routine modifies directly the matrix!
	'''
	cdef int n = A.shape[0]
	retval = c_cholesky(&A[0,0], n)
	if not retval == 0: raiseError('Problems computing Cholesky factorization!')
	return np.asarray(A)

@cr('math.vandermonde')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vandermonde(double [:] real, double [:] imag, int m, int n):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues
	'''
	cdef np.ndarray[np.complex128_t,ndim=2] Vand = np.zeros((m,n),dtype=np.complex128)
	c_vandermonde(&Vand[0,0], &real[0], &imag[0], m, n)
	return np.asarray(Vand)

@cr('math.vandermondeTime')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vandermondeTime(double [:] real, double [:] imag, int m, double [:] t):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues for a certain timesteps
	'''
	cdef int n = t.shape[0]
	cdef np.ndarray[np.complex128_t,ndim=2] Vand = np.zeros((m,n),dtype=np.complex128)
	c_vandermonde_time(&Vand[0,0], &real[0], &imag[0], m, n, &t[0])
	return np.asarray(Vand)

@cr('math.diag')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def diag(double[:,:] A):
	'''
	Returns the diagonal of A (A is a square matrix)
	'''
	cdef int m = A.shape[0]
	cdef int ii
	cdef int jj
	cdef np.ndarray[np.double_t,ndim=1] B = np.zeros((m,),dtype=np.double)
	for ii in range(m):
		for jj in range(m):
			B[ii] = A[ii][jj]
	return B

@cr('math.conj')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def conj(np.complex128_t[:,:] A):
	'''
	Returns the pointwise conjugate of A
	'''
	cdef int m = A.shape[0]
	cdef int n = A.shape[1]
	cdef int ii
	cdef int jj
	cdef np.ndarray[np.complex128_t,ndim=2] B = np.zeros((m,n),dtype=np.complex128)
	for ii in range(m):
		for jj in range(n):
			B[ii, jj] = A[ii][jj].real - A[ii][jj].imag*1j
	return B

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dinv(double[:,:] A):
	'''
	Returns the inverse of A
	'''
	retval = c_inverse(&A[0,0], A.shape[0], 'L')
	return np.asarray(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _zinv(np.complex128_t[:,:] A):
	'''
	Returns the inverse of A
	'''
	retval = c_zinverse(&A[0,0], A.shape[0], 'L')
	return np.asarray(A)

@cr('math.inv')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def inv(double_complex[:,:] A):
	'''
	Returns the inverse of A
	'''
	if double_complex is np.complex128_t:
		return _zinv(A)
	else:
		return _dinv(A)

@cr('math.flip')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def flip(double[:,:] A):
	'''
	Returns the pointwise conjugate of A
	'''
	raiseError('Function not implemented in Cython!')

@cr('math.cellCenters')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cellCenters(double[:,:] xyz,int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int ielem, icon, idim, c, cc, nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] xyz_cen = np.zeros((nel,ndim),dtype = np.double)

	for ielem in range(nel):
		# Set to zero
		for idim in range(ndim):
			xyz_cen[ielem,idim] = 0.
		cc = 0
		# Get the values of the field and the positions of the element
		for icon in range(ncon):
			c = conec[ielem,icon]
			if c < 0: break
			for idim in range(ndim):
				xyz_cen[ielem,idim] += xyz[c,idim]
			cc += 1
		# Average
		for idim in range(ndim):
			xyz_cen[ielem,idim] /= float(cc)
	return xyz_cen
