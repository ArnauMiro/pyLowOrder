#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Linear operator module
#
# Last rev: 31/08/2026

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
from libc.stdlib     cimport malloc, free
from libc.string     cimport memcpy, memset
from ..vmmath.cfuncs cimport real, real_complex, real_float, real_double, real_full
from ..vmmath.cfuncs cimport c_stranspose, c_smatmul, c_smatmulp, c_stsqr_svd, c_scompute_truncation_residual, c_scompute_truncation, c_sseparate, c_sconcatenate, c_sflip_columns
from ..vmmath.cfuncs cimport c_dtranspose, c_dmatmul, c_dmatmulp, c_dtsqr_svd, c_dcompute_truncation_residual, c_dcompute_truncation, c_dseparate, c_dconcatenate, c_dflip_columns
from ..vmmath.cfuncs cimport c_csvd, c_cdagger, c_cflip_columns
from ..vmmath.cfuncs cimport c_zsvd, c_zdagger, c_zflip_columns

from ..utils.cr       import cr, cr_start, cr_stop
from ..utils.errors   import raiseError

## Cython functions
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _slinear_operator(float[:,:] Y, float[:,:] Z, float r):
	r'''
	Reduced order version of the linear operator (Jacobian)
	'''
	cdef int my = Y.shape[0], mz = Z.shape[0], nn = Y.shape[1], retval 
	cdef int icol, irow

    # Compute SVD
	cr_start('DMD.SVD',0)
	cdef float *U_aux
	cdef float *S
	cdef float *V
	U_aux  = <float*>malloc(my*nn*sizeof(float))
	S  = <float*>malloc(nn*sizeof(float))
	V  = <float*>malloc(nn*nn*sizeof(float))
	retval = c_stsqr_svd(U_aux, S, V, &Y[0,0], my, nn)
	cr_stop('DMD.SVD',0)
	if not retval == 0: raiseError('Problems computing SVD!')

	# Remove artificial 0 rows
	cdef float *U
	U  = <float*>malloc(mz*nn*sizeof(float))
	memcpy(U, U_aux, mz*nn*sizeof(float))
	free(U_aux)

    # Truncate
	cr_start('DMD.truncate',0)
	cdef int nr

	nr = int(r) if r > 1 else c_scompute_truncation_residual(S,r,nn)
	cdef np.ndarray[np.float32_t,ndim=2] Ur   = np.zeros((mz, nr),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] Sr   = np.zeros((nr),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] Vr   = np.zeros((nr, nn),dtype=np.float32)
	c_scompute_truncation(&Ur[0,0],&Sr[0],&Vr[0,0],U,S,V,mz,nn,nn,nr)
	
	free(U)
	free(V)
	free(S)
	cr_stop('DMD.truncate',0)

    # Project Jacobian of the snapshots into the POD basis
	cr_start('DMD.linear_mapping',0)
	cdef float *aux1
	cdef float *aux2
	cdef float *Urt
	aux1   = <float*>malloc(nr*nn*sizeof(float))
	aux2   = <float*>malloc(nr*nn*sizeof(float))	
	Urt    = <float*>malloc(nr*mz*sizeof(float))
	cdef np.ndarray[np.float32_t,ndim=2] Atilde = np.zeros((nr, nr),dtype=np.float32)
	c_stranspose(&Ur[0,0], Urt, mz, nr)
	c_smatmulp(aux1, Urt, &Z[0,0], nr, nn, mz)
	for icol in range(nn):
		for irow in range(nr):
			aux2[icol*nr + irow] = Vr[irow, icol]/Sr[irow]
	c_smatmul(&Atilde[0,0], aux1, aux2, nr, nr, nn)
	free(aux1)
	free(aux2)
	free(Urt)
	cr_stop('DMD.linear_mapping',0)

	return Ur, Sr, Vr, Atilde


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _dlinear_operator(double[:,:] Y, double[:,:] Z, double r):
	r'''
	Reduced order version of the linear operator (Jacobian)
	'''
	cdef int my = Y.shape[0], mz = Z.shape[0], nn = Y.shape[1], retval 
	cdef int icol, irow

    # Compute SVD
	cr_start('DMD.SVD',0)
	cdef double *U_aux
	cdef double *S
	cdef double *V
	U_aux  = <double*>malloc(my*nn*sizeof(double))
	S  = <double*>malloc(nn*sizeof(double))
	V  = <double*>malloc(nn*nn*sizeof(double))
	retval = c_dtsqr_svd(U_aux, S, V, &Y[0,0], my, nn)
	cr_stop('DMD.SVD',0)
	if not retval == 0: raiseError('Problems computing SVD!')

	# Remove artificial 0 rows
	cdef double *U
	U  = <double*>malloc(mz*nn*sizeof(double))
	memcpy(U, U_aux, mz*nn*sizeof(double))
	free(U_aux)

    # Truncate
	cr_start('DMD.truncate',0)
	cdef int nr

	nr = int(r) if r > 1 else c_dcompute_truncation_residual(S,r,nn)
	cdef np.ndarray[np.double_t,ndim=2] Ur   = np.zeros((mz, nr),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] Sr   = np.zeros((nr),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Vr   = np.zeros((nr, nn),dtype=np.double)
	c_dcompute_truncation(&Ur[0,0],&Sr[0],&Vr[0,0],U,S,V,mz,nn,nn,nr)
	
	free(U)
	free(V)
	free(S)
	cr_stop('DMD.truncate',0)

    # Project Jacobian of the snapshots into the POD basis
	cr_start('DMD.linear_mapping',0)
	cdef double *aux1
	cdef double *aux2
	cdef double *Urt
	aux1   = <double*>malloc(nr*nn*sizeof(double))
	aux2   = <double*>malloc(nr*nn*sizeof(double))
	Urt    = <double*>malloc(nr*mz*sizeof(double))
	cdef np.ndarray[np.double_t,ndim=2] Atilde = np.zeros((nr, nr),dtype=np.double)
	c_dtranspose(&Ur[0,0], Urt, mz, nr)
	c_dmatmulp(aux1, Urt, &Z[0,0], nr, nn, mz)
	for icol in range(nn):
		for irow in range(nr):
			aux2[icol*nr + irow] = Vr[irow, icol]/Sr[irow]
	c_dmatmul(&Atilde[0,0], aux1, aux2, nr, nr, nn)
	free(aux1)
	free(aux2)
	free(Urt)
	cr_stop('DMD.linear_mapping',0)

	return Ur, Sr, Vr, Atilde

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def linear_operator(real[:,:] Y, real[:,:] Z, real r):
	r'''
	Reduced order version of the linear operator (Jacobian)

	Args:
		Y (ndarray): snapshot matrix at time t
		Z (ndarray): snapshot matrix at time t + dt
		r (int or float): maximum truncation residual
			- r > 0: target residual
			- r < 0: fraction of cumulative energy to retain

	Results:
		U (ndarray): left singular vectors of Y
		S (ndarray): singular values of Y
		VT (ndarray): right singular vectors of Y
		Atilde (ndarray): reduced-order linear operator (Jacobian)
	'''
	Y = np.ascontiguousarray(Y)
	Z = np.ascontiguousarray(Z)

	if real is double:
		return _dlinear_operator(Y,Z,r)
	else:
		return _slinear_operator(Y,Z,r)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _sconcatenate(list X, int remove_mean):
	r'''
	Joins a list of snapshot matrices for the purpose of computing the linear operator (Jacobian) from them
	'''
	cdef int ii, m, n_local, n_total, n_matrix, m_aux
	cdef int *n_list
	cdef float **X_pointer
	cdef float[:, ::1] M

	m        = X[0].shape[0]
	n_matrix = len(X)

	X_pointer = <float**>malloc(n_matrix*sizeof(float*))
	n_list    = <int*>malloc(n_matrix*sizeof(int))
	n_total   = 0
	for ii in range(n_matrix):
		M = X[ii]
		n_local = M.shape[1]
		n_total += n_local - 1
		X_pointer[ii] = &M[0,0]
		n_list[ii]    = n_local

	if m > n_total:
		m_aux = m
	else:
		m_aux = n_total
	cdef np.ndarray[np.float32_t,ndim=2] Y = np.zeros((m_aux, n_total),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] Z = np.zeros((m, n_total),dtype=np.float32)

	if remove_mean: cr_start('DMD.temporal_mean',0)
	c_sconcatenate(&Y[0,0], &Z[0,0], X_pointer, n_list, n_matrix, m, n_total, remove_mean)
	if remove_mean: cr_stop('DMD.temporal_mean',0)

	free(X_pointer)
	free(n_list)

	return Y, Z

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _dconcatenate(list X, int remove_mean):
	r'''
	Joins a list of snapshot matrices for the purpose of computing the linear operator (Jacobian) from them
	'''
	cdef int ii, m, n_local, n_total, n_matrix, m_aux
	cdef int *n_list
	cdef double **X_pointer
	cdef double[:, ::1] M

	m        = X[0].shape[0]
	n_matrix = len(X)

	X_pointer = <double**>malloc(n_matrix*sizeof(double*))
	n_list    = <int*>malloc(n_matrix*sizeof(int))
	n_total   = 0
	for ii in range(n_matrix):
		M = X[ii]
		n_local = M.shape[1]
		n_total += n_local - 1
		X_pointer[ii] = &M[0,0]
		n_list[ii]    = n_local

	if m > n_total:
		m_aux = m
	else:
		m_aux = n_total
	cdef np.ndarray[np.double_t,ndim=2] Y = np.zeros((m_aux, n_total),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Z = np.zeros((m, n_total),dtype=np.double)

	if remove_mean: cr_start('DMD.temporal_mean',0)
	c_dconcatenate(&Y[0,0], &Z[0,0], X_pointer, n_list, n_matrix, m, n_total, remove_mean)
	if remove_mean: cr_stop('DMD.temporal_mean',0)

	free(X_pointer)
	free(n_list)

	return Y, Z

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def concatenate(list X, remove_mean=False):
	r'''
	Joins a list of snapshot matrices for the purpose of computing the linear operator (Jacobian) from them

	Args:
		X (list of ndarray): list of snapshot matrices
		remove_mean (bool): whether to remove the temporal mean from each snapshot matrix

	Results:
		Y (ndarray): snapshot matrix at time t
		Z (ndarray): snapshot matrix at time t + dt
	'''
	X = [np.ascontiguousarray(M) for M in X]

	if X[0].dtype == np.double:
		return _dconcatenate(X,remove_mean)
	else:
		return _sconcatenate(X,remove_mean)


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _sseparate(float[:,:] X, int remove_mean):
	r'''
	Separates a snapshot matrix into two matrices for the purpose of computing the linear operator (Jacobian) from them
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef int m_aux

	if m > (n-1):
		m_aux = m
	else:
		m_aux = n-1
	cdef np.ndarray[np.float32_t,ndim=2] Y = np.zeros((m_aux, n-1),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] Z = np.zeros((m, n-1),dtype=np.float32)

	if remove_mean: cr_start('DMD.temporal_mean',0)
	c_sseparate(&Y[0,0], &Z[0,0], &X[0,0], m, n, remove_mean)
	if remove_mean: cr_stop('DMD.temporal_mean',0)

	return Y, Z

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _dseparate(double[:,:] X, int remove_mean):
	r'''
	Separates a snapshot matrix into two matrices for the purpose of computing the linear operator (Jacobian) from them
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef int m_aux

	if m > (n-1):
		m_aux = m
	else:
		m_aux = n-1
	cdef np.ndarray[np.double_t,ndim=2] Y = np.zeros((m_aux, n-1),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Z = np.zeros((m, n-1),dtype=np.double)

	if remove_mean: cr_start('DMD.temporal_mean',0)
	c_dseparate(&Y[0,0], &Z[0,0], &X[0,0], m, n, remove_mean)
	if remove_mean: cr_stop('DMD.temporal_mean',0)

	return Y, Z

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def separate(real[:,:] X, remove_mean=False):
	r'''
	Separates a snapshot matrix into two matrices for the purpose of computing the linear operator (Jacobian) from them

	Args:
		X (ndarray): snapshot matrix
		remove_mean (bool): whether to remove the temporal mean from the snapshot matrix

	Results:
		Y (ndarray): snapshot matrix at time t
		Z (ndarray): snapshot matrix at time t + dt
	'''
	X = np.ascontiguousarray(X)

	if real is double:
		return _dseparate(X,remove_mean)
	else:
		return _sseparate(X,remove_mean)


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _cresolvent(float[:,:] A, np.complex64_t w):
	r'''
	Resolvent of a matrix A at a given frequency w
	'''
	cdef int ii, jj, n
	n = A.shape[0]

	cdef np.complex64_t *H_inv
	H_inv  = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	
	for ii in range(n):
		for jj in range(n):
			if ii == jj:
				H_inv[ii*n + jj] = w - A[ii, jj]
			else:
				H_inv[ii*n + jj] = -A[ii, jj] 	

	cdef np.complex64_t *UT_flip
	cdef np.complex64_t *V_flip
	cdef np.complex64_t *U_flip
	cdef float *S_inv
	UT_flip  = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	V_flip   = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	U_flip   = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	S_inv    = <float*>malloc(n*sizeof(float))
	cdef np.ndarray[np.float32_t,ndim=1] S       = np.zeros((n),dtype=np.float32)
	c_csvd(V_flip, S_inv, UT_flip, H_inv, n, n)
	free(H_inv)

	for ii in range(n):
		S[ii] = 1 / S_inv[n-1-ii]
	free(S_inv)

	c_cdagger(UT_flip, U_flip, n, n)
	free(UT_flip)

	cdef np.ndarray[np.complex64_t,ndim=2] V   = np.zeros((n, n),dtype=np.complex64)
	cdef np.ndarray[np.complex64_t,ndim=2] U   = np.zeros((n, n),dtype=np.complex64)
	c_cflip_columns(U_flip, &U[0,0], n, n)
	c_cflip_columns(V_flip, &V[0,0], n, n)
	free(U_flip)
	free(V_flip)
		
	return U, S, V

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _zresolvent(double[:,:] A, np.complex128_t w):
	r'''
	Resolvent of a matrix A at a given frequency w
	'''
	cdef int ii, jj, n
	n = A.shape[0]

	cdef np.complex128_t *H_inv
	H_inv  = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	
	for ii in range(n):
		for jj in range(n):
			if ii == jj:
				H_inv[ii*n + jj] = w - A[ii, jj]
			else:
				H_inv[ii*n + jj] = -A[ii, jj] 	

	cdef np.complex128_t *UT_flip
	cdef np.complex128_t *V_flip
	cdef np.complex128_t *U_flip
	cdef double *S_inv
	UT_flip  = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	V_flip   = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	U_flip   = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	S_inv    = <double*>malloc(n*sizeof(double))
	cdef np.ndarray[np.double_t,ndim=1] S       = np.zeros((n),dtype=np.double)
	c_zsvd(V_flip, S_inv, UT_flip, H_inv, n, n)
	free(H_inv)

	for ii in range(n):
		S[ii] = 1 / S_inv[n-1-ii]
	free(S_inv)

	c_zdagger(UT_flip, U_flip, n, n)
	free(UT_flip)

	cdef np.ndarray[np.complex128_t,ndim=2] V   = np.zeros((n, n),dtype=np.complex128)
	cdef np.ndarray[np.complex128_t,ndim=2] U   = np.zeros((n, n),dtype=np.complex128)
	c_zflip_columns(U_flip, &U[0,0], n, n)
	c_zflip_columns(V_flip, &V[0,0], n, n)
	free(U_flip)
	free(V_flip)

	return U, S, V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def resolvent(real[:,:] A, object w):
	r'''
	Resolvent of a matrix A at a given frequency w

	Args:
		A (ndarray): the matrix for which to compute the resolvent
		w (int or float): the frequency at which to compute the resolvent

	Returns:
		U (ndarray): response modes
		S (ndarray): energy gains
		V (ndarray): forcing modes
	'''
	A = np.ascontiguousarray(A)

	if real is double:
		return _zresolvent(A,np.complex128(w))
	else:
		return _cresolvent(A,np.complex64(w))


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _sflip_columns(float[:,:] A):
	r'''
	Flips the columns of a matrix A in a reverse order
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] B = np.zeros((m,n),dtype=np.float32)
	c_sflip_columns(&A[0,0], &B[0,0], m,n)
	return B

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dflip_columns(double[:,:] A):
	r'''
	Flips the columns of a matrix A in a reverse order
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] B = np.zeros((m,n),dtype=np.double)
	c_dflip_columns(&A[0,0], &B[0,0], m,n)
	return B

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex64_t,ndim=2] _cflip_columns(np.complex64_t[:,:] A):
	r'''
	Flips the columns of a matrix A in a reverse order
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.complex64_t,ndim=2] B = np.zeros((m,n),dtype=np.complex64)
	c_cflip_columns(&A[0,0], &B[0,0], m,n)
	return B

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.complex128_t,ndim=2] _zflip_columns(np.complex128_t[:,:] A):
	r'''
	Flips the columns of a matrix A in a reverse order
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.complex128_t,ndim=2] B = np.zeros((m,n),dtype=np.complex128)
	c_zflip_columns(&A[0,0], &B[0,0], m,n)
	return B

@cr('math.flip_columns')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def flip_columns(real_full[:,:] A):
	r'''
	Flips the columns of a matrix A in a reverse order

	Args:
		A (ndarray): the matrix whose columns to flip

	Returns:
		B (ndarray): the matrix with flipped columns
	'''
	cdef object B
	A = np.ascontiguousarray(A)
	if real_full is np.complex128_t:
		B = _zflip_columns(A)
	elif real_full is np.complex64_t:
		B = _cflip_columns(A)
	elif real_full is double:
		B = _dflip_columns(A)
	else:
		B = _sflip_columns(A)
	return B