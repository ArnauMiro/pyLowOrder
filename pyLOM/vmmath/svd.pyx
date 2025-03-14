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
from .cfuncs       cimport c_sqr, c_ssvd, c_stsqr, c_stsqr_svd, c_srandomized_qr, c_sinit_randomized_qr, c_supdate_randomized_qr, c_srandomized_svd
from .cfuncs       cimport c_dqr, c_dsvd, c_dtsqr, c_dtsqr_svd, c_drandomized_qr, c_dinit_randomized_qr, c_dupdate_randomized_qr, c_drandomized_svd
from .cfuncs       cimport c_cqr, c_csvd, c_ctsqr, c_ctsqr_svd
from .cfuncs       cimport c_zqr, c_zsvd, c_ztsqr, c_ztsqr_svd
from ..utils.cr     import cr
from ..utils.errors import raiseError


@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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

@cython.initializedcheck(False)
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
@cython.initializedcheck(False)
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
def _stsqr(float[:,:] A):
	'''
	Parallel QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] Qi = np.zeros((m,n),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] R  = np.zeros((n,n),dtype=np.float32)
	# Compute SVD using TSQR algorithm
	retval = c_stsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing TSQR!')
	return Qi,R

@cython.initializedcheck(False)
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
	cdef np.ndarray[np.double_t,ndim=2] Qi = np.zeros((m,n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] R  = np.zeros((n,n),dtype=np.double)
	# Compute SVD using TSQR algorithm
	retval = c_dtsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing TSQR!')
	return Qi,R

@cython.initializedcheck(False)
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
	cdef np.ndarray[np.complex64_t,ndim=2] Qi = np.zeros((m,n),dtype=np.complex64)
	cdef np.ndarray[np.complex64_t,ndim=2] R  = np.zeros((n,n),dtype=np.complex64)
	# Compute SVD using TSQR algorithm
	retval = c_ctsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing TSQR!')
	return Qi,R

@cython.initializedcheck(False)
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
	cdef np.ndarray[np.complex128_t,ndim=2] Qi = np.zeros((m,n),dtype=np.complex128)
	cdef np.ndarray[np.complex128_t,ndim=2] R  = np.zeros((n,n),dtype=np.complex128)
	# Compute SVD using TSQR algorithm
	retval = c_ztsqr(&Qi[0,0],&R[0,0],&A[0,0],m,n)
	if not retval == 0: raiseError('Problems computing TSQR!')
	return Qi,R

@cr('math.tsqr')
@cython.initializedcheck(False)
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
	cdef np.ndarray[np.float32_t,ndim=1]     S = np.zeros((mn,) ,dtype=np.float32)
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
def _srandomized_qr(float[:,:] A, int r, int q, int seed):
	'''
	Parallel Randomized QR factorization using Lapack.
		Q(m,r)   
		B(r,n)  
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] Q = np.zeros((m,r),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] B = np.zeros((r,n),dtype=np.float32)
	# Compute SVD using randomized algorithm
	retval = c_srandomized_qr(&Q[0,0],&B[0,0],&A[0,0],m,n,r,q,seed)
	if not retval == 0: raiseError('Problems computing Randomized SVD!')
	return Q,B

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _drandomized_qr(double[:,:] A, int r, int q, int seed):
	'''
	Parallel Randomized QR factorization using Lapack.
		Q(m,r)   
		B(n,n)    
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] Q = np.zeros((m,r),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] B = np.zeros((r,n),dtype=np.double)
	# Compute SVD using randomized algorithm
	retval = c_drandomized_qr(&Q[0,0],&B[0,0],&A[0,0],m,n,r,q,seed)
	if not retval == 0: raiseError('Problems computing Randomized SVD!')
	return Q,B

@cr('math.randomized_qr')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def randomized_qr(real[:,:] A, const int r, const int q, const int seed=-1):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		Q(m,r)   
		B(n,r)   
	'''
	seed = <int>time(NULL) if seed < 0 else seed
	if real is double:
		return _drandomized_qr(A,r,q,seed)
	else:
		return _srandomized_qr(A,r,q,seed)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _sinit_qr_streaming(float[:,:] A, int r, int q, int seed):
	'''
	Parallel Randomized QR factorization using Lapack.
		Q(m,r)   
		B(r,n)  
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] Q = np.zeros((m,r),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] Y = np.zeros((m,r),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] B = np.zeros((r,n),dtype=np.float32)
	# Compute SVD using randomized algorithm
	retval = c_sinit_randomized_qr(&Q[0,0],&B[0,0],&Y[0,0],&A[0,0],m,n,r,q,seed)
	if not retval == 0: raiseError('Problems computing Randomized SVD!')
	return Q,B,Y

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dinit_qr_streaming(double[:,:] A, int r, int q, int seed):
	'''
	Parallel Randomized QR factorization using Lapack.
		Q(m,r)   
		B(n,n)    
	'''
	cdef int retval
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] Q = np.zeros((m,r),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Y = np.zeros((m,r),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] B = np.zeros((r,n),dtype=np.double)
	# Compute SVD using randomized algorithm
	retval = c_dinit_randomized_qr(&Q[0,0],&B[0,0],&Y[0,0],&A[0,0],m,n,r,q,seed)
	if not retval == 0: raiseError('Problems computing Randomized SVD!')
	return Q,B,Y

@cr('math.init_qr_streaming')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def init_qr_streaming(real[:,:] A, const int r, const int q, seed=None):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		Q(m,r)   
		B(n,r)   
	'''
	cdef unsigned int seed2 = <int>time(NULL) if seed == None else int(seed)
	if real is double:
		return _dinit_qr_streaming(A,r,q,seed2)
	else:
		return _sinit_qr_streaming(A,r,q,seed2)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _supdate_qr_streaming(float[:,:] Q1, float[:,:] B1, float[:,:] Yo, float[:,:] A, int r, int q, int seed):
	'''
	Parallel Randomized QR factorization using Lapack.
		Q(m,r)   
		B(r,n)  
	'''
	cdef int retval
	cdef int m  = A.shape[0], n = A.shape[1], n1 = B1.shape[1]
	cdef int n2 = n1+n
	cdef np.ndarray[np.float32_t,ndim=2] Q2 = np.zeros((m,r), dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] B2 = np.zeros((r,n2),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] Yn = np.zeros((m,r), dtype=np.float32)
	# Compute SVD using randomized algorithm
	retval = c_supdate_randomized_qr(&Q2[0,0],&B2[0,0],&Yn[0,0],&Q1[0,0],&B1[0,0],&Yo[0,0],&A[0,0],m,n,n1,n2,r,q,seed)
	if not retval == 0: raiseError('Problems updating randomized QR!')
	return Q2,B2,Yn

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dupdate_qr_streaming(double[:,:] Q1, double[:,:] B1, double[:,:] Yo, double[:,:] A, int r, int q, int seed):
	'''
	Parallel Randomized QR factorization using Lapack.
		Q(m,r)   
		B(r,n)  
	'''
	cdef int retval
	cdef int m  = A.shape[0], n = A.shape[1], n1 = B1.shape[1]
	cdef int n2 = n1+n
	cdef np.ndarray[np.double_t,ndim=2] Q2 = np.zeros((m,r), dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] B2 = np.zeros((r,n2),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Yn = np.zeros((m,r), dtype=np.double)
	# Compute SVD using randomized algorithm
	retval = c_dupdate_randomized_qr(&Q2[0,0],&B2[0,0],&Yn[0,0],&Q1[0,0],&B1[0,0],&Yo[0,0],&A[0,0],m,n,n1,n2,r,q,seed)
	if not retval == 0: raiseError('Problems updating randomized QR!')
	return Q2,B2,Yn

@cr('math.qr_iteration')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def update_qr_streaming(real[:,:] A, real [:,:] Q1, real[:,:] B1, real[:,:] Yo, const int r, const int q):
	'''
	Parallel Single value decomposition (SVD) using Lapack.
		Q(m,r)   
		B(n,r)   
	'''
	cdef unsigned int seed = <int>time(NULL)
	if real is double:
		return _dupdate_qr_streaming(Q1,B1,Yo,A,r,q,seed)
	else:
		return _supdate_qr_streaming(Q1,B1,Yo,A,r,q,seed)

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