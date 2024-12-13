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

from libc.stdlib   cimport malloc, free
from libc.string   cimport memcpy, memset
from libc.math     cimport sqrt, atan2
from libc.time     cimport time

# Fix as Open MPI does not support MPI-4 yet, and there is no nice way that I know to automatically adjust Cython to missing stuff in C header files.
# Source: https://github.com/mpi4py/mpi4py/issues/525
cdef extern from *:
	"""
	#include <mpi.h>
	
	#if (MPI_VERSION < 3) && !defined(PyMPI_HAVE_MPI_Message)
	typedef void *PyMPI_MPI_Message;
	#define MPI_Message PyMPI_MPI_Message
	#endif
	
	#if (MPI_VERSION < 4) && !defined(PyMPI_HAVE_MPI_Session)
	typedef void *PyMPI_MPI_Session;
	#define MPI_Session PyMPI_MPI_Session
	#endif
	"""
from mpi4py.libmpi cimport MPI_Comm
from mpi4py        cimport MPI
from mpi4py         import MPI

from ..utils.cr     import cr
from ..utils.errors import raiseError


## Expose C functions
cdef extern from "vector_matrix.h" nogil:
	# Single precision
	cdef void   c_stranspose        "stranspose"(float *A, float *B, const int m, const int n)
	cdef float  c_svector_norm      "svector_norm"(float *v, int start, int n)
	cdef void   c_smatmult          "smatmult"(float *C, float *A, float *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_smatmul           "smatmul"(float *C, float *A, float *B, const int m, const int n, const int k)
	cdef void   c_smatmulp          "smatmulp"(float *C, float *A, float *B, const int m, const int n, const int k)
	cdef void   c_svecmat           "svecmat"(float *v, float *A, const int m, const int n)
	cdef int    c_sinverse          "sinverse"(float *A, int N, char *UoL)
	cdef float  c_sRMSE             "sRMSE"(float *A, float *B, const int m, const int n, MPI_Comm comm)
	cdef void   c_ssort             "ssort"(float *v, int *index, int n)
	# Double precision
	cdef void   c_dtranspose        "dtranspose"(double *A, double *B, const int m, const int n)
	cdef double c_dvector_norm      "dvector_norm"(double *v, int start, int n)
	cdef void   c_dmatmult          "dmatmult"(double *C, double *A, double *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_dmatmul           "dmatmul"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_dmatmulp          "dmatmulp"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_dvecmat           "dvecmat"(double *v, double *A, const int m, const int n)
	cdef int    c_dinverse          "dinverse"(double *A, int N, char *UoL)
	cdef double c_dRMSE             "dRMSE"(double *A, double *B, const int m, const int n, MPI_Comm comm)
	cdef void   c_dsort             "dsort"(double *v, int *index, int n)
	# Single complex precision
	cdef void   c_cmatmult          "cmatmult"(np.complex64_t *C, np.complex64_t *A, np.complex64_t *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_cmatmul           "cmatmul"(np.complex64_t *C, np.complex64_t *A, np.complex64_t *B, const int m, const int n, const int k)
	cdef void 	c_cmatmulp          "cmatmulp"(np.complex64_t *C, np.complex64_t *A, np.complex64_t *B, const int m, const int n, const int k)
	cdef void   c_cvecmat           "cvecmat"(np.complex64_t *v, np.complex64_t *A, const int m, const int n)
	cdef int    c_cinverse          "cinverse"(np.complex64_t *A, int N, char *UoL)
	cdef int    c_ceigen            "ceigen"(float *real, float *imag, np.complex64_t *vecs, float *A, const int m, const int n)
	cdef int    c_ccholesky         "ccholesky"(np.complex64_t *A, int N)
	cdef void   c_cvandermonde      "cvandermonde"(np.complex64_t *Vand, float *real, float *imag, int m, int n)
	cdef void   c_cvandermonde_time "cvandermondeTime"(np.complex64_t *Vand, float *real, float *imag, int m, int n, float* t)
	cdef void   c_csort             "csort"(np.complex64_t *v, int *index, int n)
	# Double complex precision
	cdef void   c_zmatmult          "zmatmult"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_zmatmul           "zmatmul"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k)
	cdef void 	c_zmatmulp          "zmatmulp"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k)
	cdef void   c_zvecmat           "zvecmat"(np.complex128_t *v, np.complex128_t *A, const int m, const int n)
	cdef int    c_zinverse          "zinverse"(np.complex128_t *A, int N, char *UoL)
	cdef int    c_zeigen            "zeigen"(double *real, double *imag, np.complex128_t *vecs, double *A, const int m, const int n)
	cdef int    c_zcholesky         "zcholesky"(np.complex128_t *A, int N)
	cdef void   c_zvandermonde      "zvandermonde"(np.complex128_t *Vand, double *real, double *imag, int m, int n)
	cdef void   c_zvandermonde_time "zvandermondeTime"(np.complex128_t *Vand, double *real, double *imag, int m, int n, double* t)
	cdef void   c_zsort             "zsort"(np.complex128_t *v, int *index, int n)
cdef extern from "averaging.h":
	# Single precision
	cdef void c_stemporal_mean "stemporal_mean"(float *out, float *X, const int m, const int n)
	cdef void c_ssubtract_mean "ssubtract_mean"(float *out, float *X, float *X_mean, const int m, const int n)
	# Double precision
	cdef void c_dtemporal_mean "dtemporal_mean"(double *out, double *X, const int m, const int n)
	cdef void c_dsubtract_mean "dsubtract_mean"(double *out, double *X, double *X_mean, const int m, const int n)
cdef extern from "svd.h":
	# Single precision
	cdef int c_sqr             "sqr"            (float *Q,  float *R, float *A,  const int m, const int n)
	cdef int c_ssvd            "ssvd"           (float *U,  float *S, float *V,  float *Y,    const int m, const int n)
	cdef int c_stsqr           "stsqr"          (float *Qi, float *R, float *Ai, const int m, const int n, MPI_Comm comm)
	cdef int c_stsqr_svd       "stsqr_svd"      (float *Ui, float *S, float *VT, float *Ai,   const int m, const int n, MPI_Comm comm)
	cdef int c_srandomized_svd "srandomized_svd"(float *Ui, float *S, float *VT, float *Ai,   const int m, const int n, const int r, const int q, unsigned int seed, MPI_Comm comm)
	# Double precision
	cdef int c_dqr             "dqr"            (double *Q,  double *R, double *A,  const int m, const int n)
	cdef int c_dsvd            "dsvd"           (double *U,  double *S, double *V,  double *Y,   const int m, const int n)
	cdef int c_dtsqr           "dtsqr"          (double *Qi, double *R, double *Ai, const int m, const int n, MPI_Comm comm)
	cdef int c_dtsqr_svd       "dtsqr_svd"      (double *Ui, double *S, double *VT, double *Ai,  const int m, const int n, MPI_Comm comm)
	cdef int c_drandomized_svd "drandomized_svd"(double *Ui, double *S, double *VT, double *Ai,  const int m, const int n, const int r, const int q, unsigned int seed, MPI_Comm comm)
	# Single complex precision
	cdef int c_cqr        "cqr"      (np.complex64_t *Q,  np.complex64_t *R, np.complex64_t *A,  const int m,        const int n)
	cdef int c_csvd       "csvd"     (np.complex64_t *U,  float *S,          np.complex64_t *V,  np.complex64_t *Y,  const int m, const int n)
	cdef int c_ctsqr      "ctsqr"    (np.complex64_t *Qi, np.complex64_t *R, np.complex64_t *Ai, const int m,        const int n, MPI_Comm comm)
	cdef int c_ctsqr_svd  "ctsqr_svd"(np.complex64_t *Ui, float *S,          np.complex64_t *VT, np.complex64_t *Ai, const int m, const int n, MPI_Comm comm)
	# Double complex precision
	cdef int c_zqr        "zqr"      (np.complex128_t *Q,  np.complex128_t *R, np.complex128_t *A,  const int m,         const int n)
	cdef int c_zsvd       "zsvd"     (np.complex128_t *U,  double *S,          np.complex128_t *V,  np.complex128_t *Y,  const int m, const int n)
	cdef int c_ztsqr      "ztsqr"    (np.complex128_t *Qi, np.complex128_t *R, np.complex128_t *Ai, const int m,         const int n, MPI_Comm comm)
	cdef int c_ztsqr_svd  "ztsqr_svd"(np.complex128_t *Ui, double *S,          np.complex128_t *VT, np.complex128_t *Ai, const int m, const int n, MPI_Comm comm)
cdef extern from "fft.h":
	cdef int USE_FFTW3 "_USE_FFTW3"
	# Single precision
	cdef void c_sfft  "sfft" (float *psd, float *y, const float dt, const int n)
	cdef void c_snfft "snfft"(float *psd, float *t, float* y,       const int n)
	# Double precision
	cdef void c_dfft  "dfft" (double *psd, double *y, const double dt, const int n)
	cdef void c_dnfft "dnfft"(double *psd, double *t, double* y,       const int n)


## Fused type between double and complex
ctypedef fused real:
	float
	double
ctypedef fused real_complex:
	np.complex64_t
	np.complex128_t
ctypedef fused real_full:
	float
	double
	np.complex64_t
	np.complex128_t


## Cython functions
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _stranspose(float[:,:] A):
	'''
	Transposed of matrix A
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] At = np.zeros((n,m),dtype=np.float32)
	c_stranspose(&A[0,0], &At[0,0], m,n)
	return At

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dtranspose(double[:,:] A):
	'''
	Transposed of matrix A
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] At = np.zeros((n,m),dtype=np.double)
	c_dtranspose(&A[0,0], &At[0,0], m,n)
	return At

@cr('math.transpose')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def transpose(real[:,:] A):
	'''
	Transposed of matrix A
	'''
	if real is double:
		return _dtranspose(A)
	else:
		return _stranspose(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef float _svector_norm(float[:] v, int start=0):
	'''
	L2 norm of a vector
	'''
	cdef int n = v.shape[0]
	cdef float norm = 0.
	norm = c_svector_norm(&v[0],start,n)
	return norm

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef double _dvector_norm(double[:] v, int start=0):
	'''
	L2 norm of a vector
	'''
	cdef int n = v.shape[0]
	cdef double norm = 0.
	norm = c_dvector_norm(&v[0],start,n)
	return norm

@cr('math.vector_norm')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vector_norm(real[:] v, int start=0):
	'''
	L2 norm of a vector
	'''
	if real is double:
		return _dvector_norm(v,start)
	else:
		return _svector_norm(v,start)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _smatmul(float[:,:] A, float[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	cdef int m = A.shape[0], k = A.shape[1], n = B.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] C = np.zeros((m,n),dtype=np.float32)
	c_smatmul(&C[0,0],&A[0,0],&B[0,0],m,n,k)
	return C

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
	c_dmatmul(&C[0,0],&A[0,0],&B[0,0],m,n,k)
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _cmatmul(np.complex64_t[:,:] A, np.complex64_t[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	cdef int m = A.shape[0], k = A.shape[1], n = B.shape[1]
	cdef np.ndarray[np.complex64_t,ndim=2] C = np.zeros((m,n),dtype=np.complex64)
	c_cmatmul(&C[0,0],&A[0,0],&B[0,0],m,n,k)
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
def matmul(real_full[:,:] A, real_full[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	# Select type
	if real_full is np.complex128_t:
		return _zmatmul(A,B)
	elif real_full is np.complex64_t:
		return _cmatmul(A,B)
	elif real_full is double:
		return _dmatmul(A,B)
	else:
		return _smatmul(A,B)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _smatmulp(float[:,:] A, float[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	cdef int m = A.shape[0], k = A.shape[1], n = B.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] C = np.zeros((m,n),dtype=np.float32)
	c_smatmulp(&C[0,0],&A[0,0],&B[0,0],m,n,k)
	return C

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
	c_dmatmulp(&C[0,0],&A[0,0],&B[0,0],m,n,k)
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _cmatmulp(np.complex64_t[:,:] A, np.complex64_t[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	cdef int m = A.shape[0], k = A.shape[1], n = B.shape[1]
	cdef np.ndarray[np.complex64_t,ndim=2] C = np.zeros((m,n),dtype=np.complex64)
	c_cmatmulp(&C[0,0],&A[0,0],&B[0,0],m,n,k)
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
def matmulp(real_full[:,:] A, real_full[:,:] B):
	'''
	Matrix multiplication C = A x B
	'''
	if real_full is np.complex128_t:
		return _zmatmulp(A,B)
	elif real_full is np.complex64_t:
		return _cmatmulp(A,B)
	elif real_full is double:
		return _dmatmulp(A,B)
	else:
		return _smatmulp(A,B)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _svecmat(float[:] v, float[:,:] A):
	'''
	Vector times a matrix C = v x A
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] C = np.zeros((m,n),dtype=np.float32)
	memcpy(&C[0,0],&A[0,0],m*n*sizeof(float))
	c_svecmat(&v[0],&C[0,0],m,n)
	return C

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
	c_dvecmat(&v[0],&C[0,0],m,n)
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _cvecmat(np.complex64_t[:] v, np.complex64_t[:,:] A):
	'''
	Vector times a matrix C = v x A
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.complex64_t,ndim=2] C = np.zeros((m,n),dtype=np.complex64)
	memcpy(&C[0,0],&A[0,0],m*n*sizeof(np.complex64_t))
	c_cvecmat(&v[0],&C[0,0],m,n)
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
def vecmat(real_full[:] v, real_full[:,:] A):
	'''
	Vector times a matrix C = v x A
	'''
	if real_full is np.complex128_t:
		return _zvecmat(v,A)
	elif real_full is np.complex64_t:
		return _cvecmat(v,A)
	elif real_full is double:
		return _dvecmat(v,A)
	else:
		return _svecmat(v,A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _ssort(float[:] v, int[:] index):
	'''
	Sort a vector
	'''
	cdef int n = v.shape[0]
	c_ssort(&v[0],&index[0],n)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _dsort(double[:] v, int[:] index):
	'''
	Sort a vector
	'''
	cdef int n = v.shape[0]
	c_dsort(&v[0],&index[0],n)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void _csort(np.complex64_t[:] v, int[:] index):
	'''
	Sort a vector
	'''
	cdef int n = v.shape[0]
	c_csort(&v[0],&index[0],n)

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
def argsort(real_full[:] v):
	'''
	Returns the indices that sort a vector
	'''
	cdef int n = v.shape[0]
	cdef np.ndarray[np.int32_t,ndim=1] index = np.zeros((n,),dtype=np.int32)
	if real_full is np.complex128_t:
		return _zsort(v,index)
	elif real_full is np.complex64_t:
		return _csort(v,index)
	elif real_full is double:
		return _dsort(v,index)
	else:
		return _ssort(v,index)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _ceigen(float[:,:] A):
	'''
	Eigenvalues and eigenvectors using Lapack.
		real(n)   are the real eigenvalues.
		imag(n)   are the imaginary eigenvalues.
		vecs(n,n) are the right eigenvectors.
	'''
	cdef int m = A.shape[0], n = A.shape[1], retval
	cdef np.ndarray[np.float32_t,ndim=1]   real = np.zeros((n,),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1]   imag = np.zeros((n,),dtype=np.float32)
	cdef np.ndarray[np.complex64_t,ndim=2] vecs = np.zeros((n,n),dtype=np.complex64)
	# Compute eigenvalues and eigenvectors
	retval = c_ceigen(&real[0],&imag[0],&vecs[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing eigenvalues!')
	return real,imag,vecs

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _zeigen(double[:,:] A):
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
	retval = c_zeigen(&real[0],&imag[0],&vecs[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing eigenvalues!')
	return real,imag,vecs

@cr('math.eigen')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def eigen(real[:,:] A):
	'''
	Eigenvalues and eigenvectors using Lapack.
		real(n)   are the real eigenvalues.
		imag(n)   are the imaginary eigenvalues.
		vecs(n,n) are the right eigenvectors.
	'''
	if real is double:
		return _zeigen(A)
	else:
		return _ceigen(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=1] _stemporal_mean(float[:,:] X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.float32_t,ndim=1] out = np.zeros((m,),dtype=np.float32)
	# Compute temporal mean
	c_stemporal_mean(&out[0],&X[0,0],m,n)
	# Return
	return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] _dtemporal_mean(double[:,:] X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((m,),dtype=np.double)
	# Compute temporal mean
	c_dtemporal_mean(&out[0],&X[0,0],m,n)
	# Return
	return out

@cr('math.temporal_mean')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def temporal_mean(real[:,:] X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	if real is double:
		return _dtemporal_mean(X)
	else:
		return _stemporal_mean(X)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _spolar(float[:] rreal, float[:] iimag):
	'''
	Present a complex number in its polar form given its real and imaginary part
	'''
	cdef int i, n = rreal.shape[0]
	cdef np.ndarray[np.float32_t,ndim=1] mod = np.zeros((n,),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] arg = np.zeros((n,),dtype=np.float32)
	for i in range(n):
		mod[i] = sqrt(rreal[i]*rreal[i] + iimag[i]*iimag[i])
		arg[i] = atan2(iimag[i], rreal[i])
	return mod, arg

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dpolar(double[:] rreal, double[:] iimag):
	'''
	Present a complex number in its polar form given its real and imaginary part
	'''
	cdef int i, n = rreal.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] mod = np.zeros((n,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] arg = np.zeros((n,),dtype=np.double)
	for i in range(n):
		mod[i] = sqrt(rreal[i]*rreal[i] + iimag[i]*iimag[i])
		arg[i] = atan2(iimag[i], rreal[i])
	return mod, arg

@cr('math.polar')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def polar(real[:] rreal, real[:] iimag):
	'''
	Present a complex number in its polar form given its real and imaginary part
	'''
	if real is double:
		return _dpolar(rreal,iimag)
	else:
		return _spolar(rreal,iimag)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _ssubtract_mean(float[:,:] X, float[:] X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] out = np.zeros((m,n),dtype=np.float32)
	# Compute substract temporal mean
	c_ssubtract_mean(&out[0,0],&X[0,0],&X_mean[0],m,n)
	# Return
	return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dsubtract_mean(double[:,:] X, double[:] X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] out = np.zeros((m,n),dtype=np.double)
	# Compute substract temporal mean
	c_dsubtract_mean(&out[0,0],&X[0,0],&X_mean[0],m,n)
	# Return
	return out

@cr('math.subtract_mean')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def subtract_mean(real[:,:] X, real[:] X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	if real is double:
		return _dsubtract_mean(X,X_mean)
	else:
		return _ssubtract_mean(X,X_mean)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _sqr(float[:,:] A):
	'''
	QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cdef int retval, m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] Q = np.zeros((m,n),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] R = np.zeros((n,n),dtype=np.float32)
	retval = c_sqr(&Q[0,0],&R[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing QR factorization!')
	return Q,R

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
	retval = c_dqr(&Q[0,0],&R[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing QR factorization!')
	return Q,R

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _cqr(np.complex64_t[:,:] A):
	'''
	QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cdef int retval, m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.complex64_t,ndim=2] Q = np.zeros((m,n),dtype=np.complex64)
	cdef np.ndarray[np.complex64_t,ndim=2] R = np.zeros((n,n),dtype=np.complex64)
	retval = c_cqr(&Q[0,0],&R[0,0],&A[0,0],m,n)
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
def qr(real_full[:,:] A):
	'''
	QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	if real_full is np.complex128_t:
		return _zqr(A)
	elif real_full is np.complex64_t:
		return _cqr(A)
	elif real_full is double:
		return _dqr(A)
	else:
		return _sqr(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _ssvd(float[:,:] A, int do_copy):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef float *Y_copy
	cdef np.ndarray[np.float32_t,ndim=2] U = np.zeros((m,mn),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] S = np.zeros((mn,) ,dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] V = np.zeros((n,mn),dtype=np.float32)
	# Compute SVD
	if do_copy:
		Y_copy = <float*>malloc(m*n*sizeof(float))
		memcpy(Y_copy,&A[0,0],m*n*sizeof(float))
		retval = c_ssvd(&U[0,0],&S[0],&V[0,0],Y_copy,m,n)
		free(Y_copy)
	else:
		retval = c_ssvd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dsvd(double[:,:] A, int do_copy):
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
		retval = c_dsvd(&U[0,0],&S[0],&V[0,0],Y_copy,m,n)
		free(Y_copy)
	else:
		retval = c_dsvd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _csvd(np.complex64_t[:,:] A, int do_copy):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef np.complex64_t *Y_copy
	cdef np.ndarray[np.complex64_t,ndim=2] U = np.zeros((m,mn),dtype=np.complex64)
	cdef np.ndarray[np.float32_t,ndim=1]     S = np.zeros((mn,) ,dtype=np.float32)
	cdef np.ndarray[np.complex64_t,ndim=2] V = np.zeros((n,mn),dtype=np.complex64)
	# Compute SVD
	if do_copy:
		Y_copy = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
		memcpy(Y_copy,&A[0,0],m*n*sizeof(np.complex64_t))
		retval = c_csvd(&U[0,0],&S[0],&V[0,0],Y_copy,m,n)
		free(Y_copy)
	else:
		retval = c_csvd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _zsvd(np.complex128_t[:,:] A, int do_copy):
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
	cdef np.ndarray[np.double_t,ndim=1]     S = np.zeros((mn,) ,dtype=np.double)
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
def svd(real_full[:,:] A, int do_copy=True):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	if real_full is np.complex128_t:
		return _zsvd(A,do_copy)
	elif real_full is np.complex64_t:
		return _csvd(A,do_copy)
	elif real_full is double:
		return _dsvd(A,do_copy)	
	else:
		return _ssvd(A,do_copy)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _stsqr(float[:,:] A):
	'''
	Parallel QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.float32_t,ndim=2] Qi = np.zeros((m,n),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] R  = np.zeros((n,n),dtype=np.float32)
	# Compute SVD using TSQR algorithm
	retval = c_stsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing TSQR!')
	return Qi,R

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
	retval = c_dtsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing TSQR!')
	return Qi,R

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _ctsqr(np.complex64_t[:,:] A):
	'''
	Parallel QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.complex64_t,ndim=2] Qi = np.zeros((m,n),dtype=np.complex64)
	cdef np.ndarray[np.complex64_t,ndim=2] R  = np.zeros((n,n),dtype=np.complex64)
	# Compute SVD using TSQR algorithm
	retval = c_ctsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
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
def tsqr(real_full[:,:] A):
	'''
	Parallel QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	if real_full is np.complex128_t:
		return _ztsqr(A)
	elif real_full is np.complex64_t:
		return _ctsqr(A)
	elif real_full is double:
		return _dtsqr(A)
	else:
		return _stsqr(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _stsqr_svd(float[:,:] A):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.float32_t,ndim=2] U = np.zeros((m,mn),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] S = np.zeros((mn,) ,dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] V = np.zeros((n,mn),dtype=np.float32)
	# Compute SVD using TSQR algorithm
	retval = c_stsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing TSQR SVD!')
	return U,S,V

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
	retval = c_dtsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing TSQR SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _ctsqr_svd(np.complex64_t[:,:] A):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.complex64_t,ndim=2] U = np.zeros((m,mn),dtype=np.complex64)
	cdef np.ndarray[np.float32_t,ndim=1]     S = np.zeros((mn,) ,dtype=np.float32)
	cdef np.ndarray[np.complex64_t,ndim=2] V = np.zeros((n,mn),dtype=np.complex64)
	# Compute SVD using TSQR algorithm
	retval = c_ctsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,MPI_COMM.ob_mpi)
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
	cdef np.ndarray[np.double_t,ndim=1]     S = np.zeros((mn,) ,dtype=np.double)
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
def tsqr_svd(real_full[:,:] A):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	if real_full is np.complex128_t:
		return _ztsqr_svd(A)
	elif real_full is np.complex64_t:
		return _ctsqr_svd(A)
	elif real_full is double:
		return _dtsqr_svd(A)
	else:
		return _stsqr_svd(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _srandomized_svd(float[:,:] A, int r, int q):
	'''
	Parallel Randomized Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef unsigned int seed = <int>time(NULL)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.float32_t,ndim=2] U = np.zeros((m,r),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] S = np.zeros((r,) ,dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] V = np.zeros((r,n),dtype=np.float32)
	# Compute SVD using randomized algorithm
	retval = c_srandomized_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,r,q,seed,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing Randomized SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _drandomized_svd(double[:,:] A, int r, int q):
	'''
	Parallel Randomized Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef unsigned int seed = <int>time(NULL)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,r),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((r,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((r,n),dtype=np.double)
	# Compute SVD using randomized algorithm
	retval = c_drandomized_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,r,q,seed,MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing Randomized SVD!')
	return U,S,V

@cr('math.randomized_svd')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def randomized_svd(real[:,:] A, const int r, const int q):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	if real is double:
		return _drandomized_svd(A,r,q)
	else:
		return _srandomized_svd(A,r,q)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _sfft(float[:] t, float[:] y, int equispaced):
	'''
	Compute the fft of a signal y that is sampled at a
	constant timestep. Return the frequency and PSD
	'''
	cdef int n = y.shape[0]
	cdef float ts = t[1] - t[0], k_left
	cdef np.ndarray[np.float32_t,ndim=1]     x
	cdef np.ndarray[np.complex64_t,ndim=1] yf
	cdef np.ndarray[np.float32_t,ndim=1] f   = np.zeros((n,) ,dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] PSD = np.zeros((n,) ,dtype=np.float32)
	memcpy(&f[0],&y[0],n*sizeof(float))
	if equispaced:
		c_sfft(&PSD[0],&f[0],ts,n)
	else:
		if USE_FFTW3:
			c_snfft(&PSD[0],&f[0],&t[0],n)
		else:
			import nfft
			# Compute sampling frequency
			k_left = (t.shape[0]-1.)/2.
			f      = (np.arange(t.shape[0],dtype=np.float32)-k_left)/t[n-1]
			# Compute power spectra using fft
			x   = -0.5 + np.arange(t.shape[0],dtype=np.float32)/t.shape[0]
			yf  = nfft.nfft_adjoint(x,y,len(t))
			PSD = np.real(yf*np.conj(yf))/y.shape[0]
	return f, PSD

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dfft(double[:] t, double[:] y, int equispaced):
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
		c_dfft(&PSD[0],&f[0],ts,n)
	else:
		if USE_FFTW3:
			c_dnfft(&PSD[0],&f[0],&t[0],n)
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

@cr('math.fft')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def fft(real[:] t, real[:] y, int equispaced=True):
	'''
	Compute the fft of a signal y that is sampled at a
	constant timestep. Return the frequency and PSD
	'''
	if real is double:
		return _dfft(t,y,equispaced)
	else:
		return _sfft(t,y,equispaced)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef float _sRMSE(float[:,:] A, float[:,:] B):
	'''
	Compute RMSE between X_POD and X
	'''
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef int m = A.shape[0], n = B.shape[1]
	cdef float rmse = 0.
	rmse = c_sRMSE(&A[0,0],&B[0,0],m,n,MPI_COMM.ob_mpi)
	return rmse

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef double _dRMSE(double[:,:] A, double[:,:] B):
	'''
	Compute RMSE between X_POD and X
	'''
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	cdef int m = A.shape[0], n = B.shape[1]
	cdef double rmse = 0.
	rmse = c_dRMSE(&A[0,0],&B[0,0],m,n,MPI_COMM.ob_mpi)
	return rmse

@cr('math.RMSE')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def RMSE(real[:,:] A, real[:,:] B):
	'''
	Compute RMSE between X_POD and X
	'''
	if real is double:
		return _dRMSE(A,B)
	else:
		return _sRMSE(A,B)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _ccholesky(np.complex64_t[:,:] A):
	'''
	Compute the Lower Cholesky decomposition of matrix A. The C routine modifies directly the matrix!
	'''
	cdef int n = A.shape[0]
	retval = c_ccholesky(&A[0,0], n)
	if not retval == 0: raiseError('Problems computing Cholesky factorization!')
	return np.asarray(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zcholesky(np.complex128_t[:,:] A):
	'''
	Compute the Lower Cholesky decomposition of matrix A. The C routine modifies directly the matrix!
	'''
	cdef int n = A.shape[0]
	retval = c_zcholesky(&A[0,0], n)
	if not retval == 0: raiseError('Problems computing Cholesky factorization!')
	return np.asarray(A)

@cr('math.cholesky')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cholesky(real_complex[:,:] A):
	'''
	Compute the Lower Cholesky decomposition of matrix A. The C routine modifies directly the matrix!
	'''
	if real_complex is np.complex128_t:
		return _zcholesky(A)
	else:
		return _ccholesky(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _cvandermonde(float[:] rreal, float[:] iimag, int m, int n):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues
	'''
	cdef np.ndarray[np.complex64_t,ndim=2] vand = np.zeros((m,n),dtype=np.complex64)
	c_cvandermonde(&vand[0,0], &rreal[0], &iimag[0], m, n)
	return np.asarray(vand)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zvandermonde(double[:] rreal, double[:] iimag, int m, int n):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues
	'''
	cdef np.ndarray[np.complex128_t,ndim=2] vand = np.zeros((m,n),dtype=np.complex128)
	c_zvandermonde(&vand[0,0], &rreal[0], &iimag[0], m, n)
	return np.asarray(vand)

@cr('math.vandermonde')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vandermonde(real[:] rreal, real[:] iimag, int m, int n):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues
	'''
	if real is double:
		return _zvandermonde(rreal,iimag,m,n)
	else:
		return _cvandermonde(rreal,iimag,m,n)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _cvandermondeTime(float[:] rreal, float[:] iimag, int m, float[:] t):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues for a certain timesteps
	'''
	cdef int n = t.shape[0]
	cdef np.ndarray[np.complex64_t,ndim=2] vand = np.zeros((m,n),dtype=np.complex64)
	c_cvandermonde_time(&vand[0,0], &rreal[0], &iimag[0], m, n, &t[0])
	return np.asarray(vand)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zvandermondeTime(double[:] rreal, double[:] iimag, int m, double[:] t):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues for a certain timesteps
	'''
	cdef int n = t.shape[0]
	cdef np.ndarray[np.complex128_t,ndim=2] vand = np.zeros((m,n),dtype=np.complex128)
	c_zvandermonde_time(&vand[0,0], &rreal[0], &iimag[0], m, n, &t[0])
	return np.asarray(vand)

@cr('math.vandermondeTime')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vandermondeTime(real[:] rreal, real[:] iimag, int m, real[:] t):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues for a certain timesteps
	'''
	if real is double:
		return _zvandermondeTime(rreal,iimag,m,t)
	else:
		return _cvandermondeTime(rreal,iimag,m,t)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=1] _sdiag(float[:,:] A):
	'''
	Returns the diagonal of A (A is a square matrix)
	'''
	cdef int m = A.shape[0]
	cdef int ii
	cdef int jj
	cdef np.ndarray[np.float32_t,ndim=1] B = np.zeros((m,),dtype=np.float32)
	for ii in range(m):
		for jj in range(m):
			B[ii] = A[ii][jj]
	return B

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] _ddiag(double[:,:] A):
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

@cr('math.diag')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def diag(real[:,:] A):
	'''
	Returns the diagonal of A (A is a square matrix)
	'''
	if real is double:
		return _ddiag(A)
	else:
		return _sdiag(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _cconj(np.complex64_t[:,:] A):
	'''
	Returns the pointwise conjugate of A
	'''
	cdef int m = A.shape[0]
	cdef int n = A.shape[1]
	cdef int ii
	cdef int jj
	cdef np.ndarray[np.complex64_t,ndim=2] B = np.zeros((m,n),dtype=np.complex64)
	for ii in range(m):
		for jj in range(n):
			B[ii, jj] = A[ii][jj].real - A[ii][jj].imag*1j
	return B

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zconj(np.complex128_t[:,:] A):
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

@cr('math.conj')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def conj(real_complex[:,:] A):
	'''
	Returns the pointwise conjugate of A
	'''
	if real_complex is np.complex128_t:
		return _zconj(A)
	else:
		return _cconj(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _sinv(float[:,:] A):
	'''
	Returns the inverse of A
	'''
	retval = c_sinverse(&A[0,0], A.shape[0], 'L')
	return np.asarray(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dinv(double[:,:] A):
	'''
	Returns the inverse of A
	'''
	retval = c_dinverse(&A[0,0], A.shape[0], 'L')
	return np.asarray(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _cinv(np.complex64_t[:,:] A):
	'''
	Returns the inverse of A
	'''
	retval = c_cinverse(&A[0,0], A.shape[0], 'L')
	return np.asarray(A)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zinv(np.complex128_t[:,:] A):
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
def inv(real_full[:,:] A):
	'''
	Returns the inverse of A
	'''
	if real_full is np.complex128_t:
		return _zinv(A)
	elif real_full is np.complex64_t:
		return _cinv(A)
	elif real_full is double:
		return _dinv(A)
	else:
		return _sinv(A)

@cr('math.flip')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def flip(real[:,:] A):
	'''
	Returns the pointwise conjugate of A
	'''
	raiseError('Function not implemented in Cython!')

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _scellCenters(float[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int ielem, icon, idim, c, cc, nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] xyz_cen = np.zeros((nel,ndim),dtype = np.float32)

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

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dcellCenters(double[:,:] xyz, int[:,:] conec):
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

@cr('math.cellCenters')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cellCenters(real[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	if real is double:
		return _dcellCenters(xyz,conec)
	else:
		return _scellCenters(xyz,conec)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _snormals(float[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int ielem, icon, c, cc, nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.float32_t,ndim=1] cen     = np.zeros((ndim,),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] u       = np.zeros((ndim,),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] v       = np.zeros((ndim,),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] normals = np.zeros((nel,ndim),dtype = np.float32)

	for ielem in range(nel):
		# Set to zero
		for idim in range(ndim):
			cen[idim]           = 0.
			normals[ielem,idim] = 0.
		# Compute centroid
		cc = 0
		for icon in range(ncon):
			c = conec[ielem,icon]
			if c < 0: break
			for idim in range(ndim):
				cen[idim] += xyz[c,idim]
			cc += 1
		for idim in range(ndim):
			cen[idim] /= float(cc)
		# Compute normal
		# Compute u, v
		icon = cc - 1
		c    = conec[ielem,0]
		cc   = conec[ielem,icon]
		for idim in range(ndim):
			u[idim] = xyz[c,idim]  - cen[idim]
			v[idim] = xyz[cc,idim] - cen[idim]
		# Cross product
		normals[ielem,0] += 0.5*(u[1]*v[2] - u[2]*v[1])
		normals[ielem,1] += 0.5*(u[2]*v[0] - u[0]*v[2])
		normals[ielem,2] += 0.5*(u[0]*v[1] - u[1]*v[0])
		for icon in range(1,ncon):
			c  = conec[ielem,icon]
			cc = conec[ielem,icon-1]
			if c < 0: break
			# Compute u, v
			for idim in range(ndim):
				u[idim] = xyz[c,idim]  - cen[idim]
				v[idim] = xyz[cc,idim] - cen[idim]
			# Cross product
			normals[ielem,0] += 0.5*(u[1]*v[2] - u[2]*v[1])
			normals[ielem,1] += 0.5*(u[2]*v[0] - u[0]*v[2])
			normals[ielem,2] += 0.5*(u[0]*v[1] - u[1]*v[0])

	return normals

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dnormals(double[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int ielem, icon, c, cc, nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] cen     = np.zeros((ndim,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] u       = np.zeros((ndim,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] v       = np.zeros((ndim,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] normals = np.zeros((nel,ndim),dtype = np.double)

	for ielem in range(nel):
		# Set to zero
		for idim in range(ndim):
			cen[idim]           = 0.
			normals[ielem,idim] = 0.
		# Compute centroid
		cc = 0
		for icon in range(ncon):
			c = conec[ielem,icon]
			if c < 0: break
			for idim in range(ndim):
				cen[idim] += xyz[c,idim]
			cc += 1
		for idim in range(ndim):
			cen[idim] /= float(cc)
		# Compute normal
		# Compute u, v
		icon = cc - 1
		c    = conec[ielem,0]
		cc   = conec[ielem,icon]
		for idim in range(ndim):
			u[idim] = xyz[c,idim]  - cen[idim]
			v[idim] = xyz[cc,idim] - cen[idim]
		# Cross product
		normals[ielem,0] += 0.5*(u[1]*v[2] - u[2]*v[1])
		normals[ielem,1] += 0.5*(u[2]*v[0] - u[0]*v[2])
		normals[ielem,2] += 0.5*(u[0]*v[1] - u[1]*v[0])
		for icon in range(1,ncon):
			c  = conec[ielem,icon]
			cc = conec[ielem,icon-1]
			if c < 0: break
			# Compute u, v
			for idim in range(ndim):
				u[idim] = xyz[c,idim]  - cen[idim]
				v[idim] = xyz[cc,idim] - cen[idim]
			# Cross product
			normals[ielem,0] += 0.5*(u[1]*v[2] - u[2]*v[1])
			normals[ielem,1] += 0.5*(u[2]*v[0] - u[0]*v[2])
			normals[ielem,2] += 0.5*(u[0]*v[1] - u[1]*v[0])

	return normals

@cr('math.normals')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def normals(real[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	if real is double:
		return _dnormals(xyz,conec)
	else:
		return _snormals(xyz,conec)
