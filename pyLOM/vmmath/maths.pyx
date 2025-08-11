#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - vector/matrix math.
#
# Last rev: 27/10/2021

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
from libc.stdlib   cimport malloc, free
from libc.string   cimport memcpy, memset
from libc.math     cimport sqrt, atan2
from .cfuncs       cimport real, real_complex, real_full
from .cfuncs       cimport c_stranspose, c_svector_sum, c_svector_norm, c_svector_mean, c_smatmul, c_smatmulp, c_svecmat, c_sinv, c_ssort
from .cfuncs       cimport c_dtranspose, c_dvector_sum, c_dvector_norm, c_dvector_mean, c_dmatmul, c_dmatmulp, c_dvecmat, c_dinv, c_dsort
from .cfuncs       cimport c_cmatmul, c_cmatmulp, c_cvecmat, c_cinv, c_ceigen, c_ccholesky, c_cvandermonde, c_cvandermonde_time, c_csort
from .cfuncs       cimport c_zmatmul, c_zmatmulp, c_zvecmat, c_zinv, c_zeigen, c_zcholesky, c_zvandermonde, c_zvandermonde_time, c_zsort

from ..utils.cr     import cr
from ..utils.errors import raiseError


## Cython functions
@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def transpose(real[:,:] A):
	r'''
	Transposed of matrix A

	Args:
		A (np.ndarray): Matrix to be transposed
	
	Results
		np.ndarray: Transposed matrix
	'''
	if real is double:
		return _dtranspose(A)
	else:
		return _stranspose(A)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef float _svector_sum(float[:] v, int start=0):
	'''
	Sum of a vector
	'''
	cdef int n = v.shape[0]
	cdef float norm = 0.
	norm = c_svector_sum(&v[0],start,n)
	return norm

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef double _dvector_sum(double[:] v, int start=0):
	'''
	Sum of a vector
	'''
	cdef int n = v.shape[0]
	cdef double norm = 0.
	norm = c_dvector_sum(&v[0],start,n)
	return norm

@cr('math.vector_sum')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vector_sum(real[:] v, int start=0):
	r'''
	Sum of a vector

	Args:
		v (np.ndarray): a vector
		start (int): position of the vector where to start the sum
	
	Result:
		float: sum of the vector
	'''
	if real is double:
		return _dvector_sum(v,start)
	else:
		return _svector_sum(v,start)

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vector_norm(real[:] v, int start=0):
	r'''
	L2 norm of a vector
	
	Args:
		v (np.ndarray): a vector
		start (int): position of the vector where to start the norm
	
	Result:
		float: norm of the vector
	'''
	if real is double:
		return _dvector_norm(v,start)
	else:
		return _svector_norm(v,start)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef float _svector_mean(float[:] v, int start=0):
	'''
	Sum of a vector
	'''
	cdef int n = v.shape[0]
	cdef float norm = 0.
	norm = c_svector_mean(&v[0],start,n)
	return norm

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef double _dvector_mean(double[:] v, int start=0):
	'''
	Sum of a vector
	'''
	cdef int n = v.shape[0]
	cdef double norm = 0.
	norm = c_dvector_mean(&v[0],start,n)
	return norm

@cr('math.vector_mean')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vector_mean(real[:] v, int start=0):
	r'''
	Mean of a vector

	Args:
		v (np.ndarray): a vector
		start (int): position of the vector where to start the mean
	
	Result:
		float: mean of the vector
	'''
	if real is double:
		return _dvector_mean(v,start)
	else:
		return _svector_mean(v,start)

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def matmul(real_full[:,:] A, real_full[:,:] B):
	r'''
	Matrix multiplication 
	C = A x B

	Args:
		A (np.ndarray): Matrix A (M,Q)
		B (np.ndarray): Matrix B (Q,N)
	
	Result:
		np.ndarray: Resulting matrix C (M,N)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def matmulp(real_full[:,:] A, real_full[:,:] B):
	r'''
	Matrix multiplication in parallel
	C = A x B 

	.. warning::
	A and B are distributed along the processors and C is the same for all of them

	Args:
		A (np.ndarray): Matrix A (M,Q)
		B (np.ndarray): Matrix B (Q,N)
	
	Result:
		np.ndarray: Resulting matrix C (M,N)
	'''
	if real_full is np.complex128_t:
		return _zmatmulp(A,B)
	elif real_full is np.complex64_t:
		return _cmatmulp(A,B)
	elif real_full is double:
		return _dmatmulp(A,B)
	else:
		return _smatmulp(A,B)

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vecmat(real_full[:] v, real_full[:,:] A):
	r'''
	Vector times a matrix 
	C = v x A

	Args:
		v (np.ndarray): Vector v (M,)
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: Resulting matrix C (M,N)
	'''
	if real_full is np.complex128_t:
		return _zvecmat(v,A)
	elif real_full is np.complex64_t:
		return _cvecmat(v,A)
	elif real_full is double:
		return _dvecmat(v,A)
	else:
		return _svecmat(v,A)

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def argsort(real_full[:] v):
	r'''
	Returns the indices that sort a vector

	Args:
		v (np.ndarray): Vector v (M,)
	
	Result:
		np.ndarray: Indices that sort v (M,)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def eigen(real[:,:] A):
	r'''
	Eigenvalues and eigenvectors.

	.. warning::
	GPU implementation of this algoritm is still slow and
	thus will be executed purely on CPU level until
	cupy implements linalg.eig

	Args:
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: the real eigenvalues, real(M)
		np.ndarray: the imaginary eigenvalues, imag(M)
		np.ndarray: the right eigenvectors, vecs(M,M)
	'''
	if real is double:
		return _zeigen(A)
	else:
		return _ceigen(A)

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def polar(real[:] rreal, real[:] iimag):
	r'''
	Present a complex number in its polar form given its real and imaginary part

	Args:
		real (np.ndarray): the real component
		imag (np.ndarray): the imaginary component

	Result:
		np.ndarray: the modulus
		np.ndarray: the argument
	'''
	if real is double:
		return _dpolar(rreal,iimag)
	else:
		return _spolar(rreal,iimag)

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cholesky(real_complex[:,:] A):
	r'''
	Conjugates complex number A

	Args:
		A (np.ndarray): Vector, matrix or number A
	
	Result:
		np.ndarray: Conjugate of A
	'''
	if real_complex is np.complex128_t:
		return _zcholesky(A)
	else:
		return _ccholesky(A)

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vandermonde(real[:] rreal, real[:] iimag, int m, int n):
	r'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues

	Args:
		real (np.ndarray): the real component
		imag (np.ndarray): the imaginary component
		m (int): number of rows for the Vandermode matrix
		n (int): number of columns for the Vandermode matrix

	Result:
		np.ndarray: the Vandermonde matrix
	'''
	if real is double:
		return _zvandermonde(rreal,iimag,m,n)
	else:
		return _cvandermonde(rreal,iimag,m,n)

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def vandermondeTime(real[:] rreal, real[:] iimag, int m, real[:] t):
	r'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues


	Args:
		real (np.ndarray): the real component
		imag (np.ndarray): the imaginary component
		m (int): number of rows for the Vandermode matrix
		time (np.ndarray): the time vector

	Result:
		np.ndarray: the Vandermonde matrix
	'''
	if real is double:
		return _zvandermondeTime(rreal,iimag,m,t)
	else:
		return _cvandermondeTime(rreal,iimag,m,t)

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def diag(real[:,:] A):
	r'''
	If A is a matrix it returns its diagonal, if its a vector it returns
	a diagonal matrix with A in its diagonal

	Args:
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: Diagonal of A (M,)
	'''
	if real is double:
		return _ddiag(A)
	else:
		return _sdiag(A)

@cython.initializedcheck(False)
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
			B[ii, jj] = crealf(A[ii][jj]) - cimagf(A[ii][jj])*I
	return B

@cython.initializedcheck(False)
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
			B[ii, jj] = creal(A[ii][jj]) - cimag(A[ii][jj])*J
	return B

@cr('math.conj')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def conj(real_complex[:,:] A):
	r'''
	Conjugates complex number A

	Args:
		A (np.ndarray): Vector, matrix or number A
	
	Result:
		np.ndarray: Conjugate of A
	'''
	if real_complex is np.complex128_t:
		return _zconj(A)
	else:
		return _cconj(A)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _sinv(float[:,:] A):
	'''
	Returns the inverse of A
	'''
	retval = c_sinv(&A[0,0], A.shape[0], A.shape[1])
	return np.asarray(A)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dinv(double[:,:] A):
	'''
	Returns the inverse of A
	'''
	retval = c_dinv(&A[0,0], A.shape[0], A.shape[1])
	return np.asarray(A)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _cinv(np.complex64_t[:,:] A):
	'''
	Returns the inverse of A
	'''
	retval = c_cinv(&A[0,0], A.shape[0], A.shape[1])
	return np.asarray(A)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zinv(np.complex128_t[:,:] A):
	'''
	Returns the inverse of A
	'''
	retval = c_zinv(&A[0,0], A.shape[0], A.shape[1])
	return np.asarray(A)

@cr('math.inv')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def inv(real_full[:,:] A):
	r'''
	Computes the inverse matrix of A

	Args:
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: Inverse of A (M,N)
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
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def flip(real[:,:] A):
	r'''
	Returns the pointwise conjugate of A

	.. warning::
	This function is not implemented in the
	compiled layer and will raise an error if used

	Args:
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: Flipped version of A (M,N)
	'''
	raiseError('Function not implemented in Cython!')