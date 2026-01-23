#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module.
#
# Last rev: 27/10/2021

cimport cython
cimport numpy as np

import numpy as np

from libc.time     cimport time
from libc.stdlib   cimport malloc, free
from libc.string   cimport memcpy, memset
from .cfuncs       cimport real, real_complex, real_full
from .cfuncs       cimport c_ssvd, c_stsqr_svd, c_srandomized_qr, c_slocal_randomized_qr, c_sinit_randomized_qr, c_supdate_randomized_qr, c_srandomized_svd
from .cfuncs       cimport c_dsvd, c_dtsqr_svd, c_drandomized_qr, c_dlocal_randomized_qr, c_dinit_randomized_qr, c_dupdate_randomized_qr, c_drandomized_svd
from .cfuncs       cimport c_csvd, c_ctsqr_svd
from .cfuncs       cimport c_zsvd, c_ztsqr_svd
from ..utils.cr     import cr
from ..utils.errors import raiseError


@cython.initializedcheck(False)
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
	cdef np.ndarray[np.float32_t,ndim=2] V = np.zeros((mn,n),dtype=np.float32)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
	cdef np.ndarray[np.float32_t,ndim=1]   S = np.zeros((mn,) ,dtype=np.float32)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
	cdef np.ndarray[np.float32_t,ndim=2] U = np.zeros((m,mn),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] S = np.zeros((mn,) ,dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] V = np.zeros((n,mn),dtype=np.float32)
	# Compute SVD using TSQR algorithm
	retval = c_stsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing TSQR SVD!')
	return U,S,V

@cython.initializedcheck(False)
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
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Compute SVD using TSQR algorithm
	retval = c_dtsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing TSQR SVD!')
	return U,S,V

@cython.initializedcheck(False)
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
	cdef np.ndarray[np.complex64_t,ndim=2] U = np.zeros((m,mn),dtype=np.complex64)
	cdef np.ndarray[np.float32_t,ndim=1]   S = np.zeros((mn,) ,dtype=np.float32)
	cdef np.ndarray[np.complex64_t,ndim=2] V = np.zeros((n,mn),dtype=np.complex64)
	# Compute SVD using TSQR algorithm
	retval = c_ctsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing TSQR SVD!')
	return U,S,V

@cython.initializedcheck(False)
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
	cdef np.ndarray[np.complex128_t,ndim=2] U = np.zeros((m,mn),dtype=np.complex128)
	cdef np.ndarray[np.double_t,ndim=1]     S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.complex128_t,ndim=2] V = np.zeros((n,mn),dtype=np.complex128)
	# Compute SVD using TSQR algorithm
	retval = c_ztsqr_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing TSQR SVD!')
	return U,S,V

@cr('math.tsqr_svd')
@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _srandomized_svd(float[:,:] A, int r, int q, int seed):
	'''
	Parallel Randomized Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef np.ndarray[np.float32_t,ndim=2] U = np.zeros((m,r),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] S = np.zeros((r,) ,dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] V = np.zeros((r,n),dtype=np.float32)
	# Compute SVD using randomized algorithm
	retval = c_srandomized_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,r,q,seed)
	if not retval == 0: raiseError('Problems computing Randomized SVD!')
	return U,S,V

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _drandomized_svd(double[:,:] A, int r, int q, int seed):
	'''
	Parallel Randomized Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1], mn = min(m,n)
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,r),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((r,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((r,n),dtype=np.double)
	# Compute SVD using randomized algorithm
	retval = c_drandomized_svd(&U[0,0],&S[0],&V[0,0],&A[0,0],m,n,r,q,seed)
	if not retval == 0: raiseError('Problems computing Randomized SVD!')
	return U,S,V

@cr('math.randomized_svd')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def randomized_svd(real[:,:] A, const int r, const int q, const int seed=-1):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	seed = <int>time(NULL) if seed < 0 else seed
	if real is double:
		return _drandomized_svd(A,r,q,seed)
	else:
		return _srandomized_svd(A,r,q,seed)