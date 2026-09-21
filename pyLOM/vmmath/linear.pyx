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
from ..vmmath.cfuncs cimport c_stranspose, c_smatmul, c_smatmulp, c_stsqr_svd, c_scompute_truncation_residual, c_scompute_truncation, c_stemporal_mean, c_ssubtract_mean, c_sflip_columns
from ..vmmath.cfuncs cimport c_dtranspose, c_dmatmul, c_dmatmulp, c_dtsqr_svd, c_dcompute_truncation_residual, c_dcompute_truncation, c_dtemporal_mean, c_dsubtract_mean, c_dflip_columns
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
	'''
	I dont understand mn here (old version). I think it is to avoid a m<n matrix. New method is implemented.
	'''
	cdef int my = Y.shape[0], mz = Z.shape[0], nn = Y.shape[1], n = (nn+1), retval 
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
	cdef float *aux3
	cdef float *Urt
	aux1   = <float*>malloc(nr*nn*sizeof(float))
	aux2   = <float*>malloc(nr*nn*sizeof(float))
	aux3   = <float*>malloc(nr*sizeof(float))
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
	free(aux3)
	free(Urt)
	cr_stop('DMD.linear_mapping',0)

	return Ur, Sr, Vr, Atilde


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _dlinear_operator(double[:,:] Y, double[:,:] Z, double r):
	'''
	I dont understand mn here (old version). I think it is to avoid a m<n matrix. New method is implemented.
	'''
	cdef int my = Y.shape[0], mz = Z.shape[0], nn = Y.shape[1], n = (nn+1), retval 
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

	nr = int(r) if r > 1 else c_dcompute_truncation_residual(S,r,n-1)
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
	cdef double *aux3
	cdef double *Urt
	aux1   = <double*>malloc(nr*nn*sizeof(double))
	aux2   = <double*>malloc(nr*nn*sizeof(double))
	aux3   = <double*>malloc(nr*sizeof(double))
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
	free(aux3)
	free(Urt)
	cr_stop('DMD.linear_mapping',0)

	return Ur, Sr, Vr, Atilde

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def linear_operator(real[:,:] Y, real[:,:] Z, real r):
	'''

	'''
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
	cdef int ii, jj, m, m_aux, n, ni, offset

	m = X[0].shape[0]
	n = sum([M.shape[1] - 1 for M in X])

	# Create the matrices Y, Z
	if m > n:
		m_aux = m
	else:
		m_aux = n
	cdef np.ndarray[np.float32_t,ndim=2] Y   = np.zeros((m_aux, n),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] Z   = np.zeros((m, n),dtype=np.float32)

	# Fill the matrices
	offset = 0
	cdef float *X_mean
	cdef float *X_meanless
	cdef float[:, ::1] M 
	X_mean     = <float*>malloc(m*sizeof(float))
	for ii in range(len(X)):
		M = X[ii]
		ni = M.shape[1]
		X_meanless = <float*>malloc(m*ni*sizeof(float))
		if remove_mean:
			cr_start('DMD.temporal_mean',0)
			c_stemporal_mean(X_mean,&M[0,0],m,ni)
			c_ssubtract_mean(X_meanless,&M[0,0],X_mean,m,ni)
			for jj in range(m):
				memcpy(&Y[jj, offset], &X_meanless[jj*ni], (ni-1)*sizeof(float))
				memcpy(&Z[jj, offset], &X_meanless[jj*ni+1], (ni-1)*sizeof(float))
			offset += (ni-1)
			cr_stop('DMD.temporal_mean',0)
		else:
			Y[:m,offset:offset + ni-1] = M[:,:-1]
			Z[:,offset:offset + ni-1] = M[:,1:]
			offset += (ni-1)
		free(X_meanless)
    
	free(X_mean)

	return Y, Z

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _dconcatenate(list X, int remove_mean):
	cdef int ii, jj, m, m_aux, n, ni, offset

	m = X[0].shape[0]
	n = sum([M.shape[1] - 1 for M in X])

	# Create the matrices Y, Z
	if m > n:
		m_aux = m
	else:
		m_aux = n
	cdef np.ndarray[np.double_t,ndim=2] Y   = np.zeros((m_aux, n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Z   = np.zeros((m, n),dtype=np.double)

	# Fill the matrices
	offset = 0
	cdef double *X_mean
	cdef double *X_meanless
	cdef double[:, ::1] M 
	X_mean     = <double*>malloc(m*sizeof(double))
	for ii in range(len(X)):
		M = X[ii]
		ni = M.shape[1]
		X_meanless = <double*>malloc(m*ni*sizeof(double))  
		if remove_mean:
			cr_start('DMD.temporal_mean',0)
			c_dtemporal_mean(X_mean,&M[0,0],m,ni)
			c_dsubtract_mean(X_meanless,&M[0,0],X_mean,m,ni)
			for jj in range(m):
				memcpy(&Y[jj, offset], &X_meanless[jj*ni], (ni-1)*sizeof(double))
				memcpy(&Z[jj, offset], &X_meanless[jj*ni+1], (ni-1)*sizeof(double))
			offset += (ni-1)
			cr_stop('DMD.temporal_mean',0)
		else:
			Y[:m,offset:offset + ni-1] = M[:,:-1]
			Z[:,offset:offset + ni-1] = M[:,1:]
			offset += (ni-1)
		free(X_meanless)
    	
	free(X_mean)

	return Y, Z

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def concatenate(list X, remove_mean=False):

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
	cdef int jj, m, m_aux, n

	m = X.shape[0]
	n = X.shape[1]

	# Create the matrices Y, Z
	if m > n:
		m_aux = m
	else:
		m_aux = n
	cdef np.ndarray[np.float32_t,ndim=2] Y   = np.zeros((m_aux, n),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] Z   = np.zeros((m, n),dtype=np.float32)

	# Fill the matrices
	cdef float *X_mean
	cdef float *X_meanless
	X_mean     = <float*>malloc(m*sizeof(float))
	X_meanless = <float*>malloc(m*n*sizeof(float))
	if remove_mean:
		cr_start('DMD.temporal_mean',0)
		c_stemporal_mean(X_mean,&X[0,0],m,n)
		c_ssubtract_mean(X_meanless,&X[0,0],X_mean,m,n)
		for jj in range(m):
			memcpy(&Y[jj, 0], &X_meanless[jj*n], (n-1)*sizeof(float))
			memcpy(&Z[jj, 0], &X_meanless[jj*n+1], (n-1)*sizeof(float))
		cr_stop('DMD.temporal_mean',0)
	else:
		Y[:m,:] = X[:,:-1]
		Z[:,:] = X[:,1:]

	free(X_meanless)
	free(X_mean)

	return Y, Z

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _dseparate(double[:,:] X, int remove_mean):
	cdef int jj, m, m_aux, n

	m = X.shape[0]
	n = X.shape[1]

	# Create the matrices Y, Z
	if m > n:
		m_aux = m
	else:
		m_aux = n
	cdef np.ndarray[np.double_t,ndim=2] Y   = np.zeros((m_aux, n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Z   = np.zeros((m, n),dtype=np.double)

	# Fill the matrices
	cdef double *X_mean
	cdef double *X_meanless
	X_mean     = <double*>malloc(m*sizeof(double))
	X_meanless = <double*>malloc(m*n*sizeof(double))
	if remove_mean:
		cr_start('DMD.temporal_mean',0)
		c_dtemporal_mean(X_mean,&X[0,0],m,n)
		c_dsubtract_mean(X_meanless,&X[0,0],X_mean,m,n)
		for jj in range(m):
			memcpy(&Y[jj, 0], &X_meanless[jj*n], (n-1)*sizeof(double))
			memcpy(&Z[jj, 0], &X_meanless[jj*n+1], (n-1)*sizeof(double))
		cr_stop('DMD.temporal_mean',0)
	else:
		Y[:m,:] = X[:,:-1]
		Z[:,:] = X[:,1:]

	free(X_meanless)
	free(X_mean)

	return Y, Z


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def separate(real[:,:] X, remove_mean=False):

	if real is double:
		return _dseparate(X,remove_mean)
	else:
		return _sseparate(X,remove_mean)


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef tuple _cresolvent(float[:,:] A, np.complex64_t f):

	cdef int ii, jj, n
	n = A.shape[0]

	cdef np.complex64_t *H_inv
	H_inv  = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	
	# COMPTE ELS SIGNES!
	for ii in range(n):
		for jj in range(n):
			if ii == jj:
				H_inv[ii*n + jj] = f - A[ii, jj]
			else:
				H_inv[ii*n + jj] = -A[ii, jj] 	

	cdef np.complex64_t *UT_flip
	cdef np.complex64_t *V_flip
	cdef np.complex64_t *U_flip
	cdef float *S_inv
	UT_flip  = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	V_flip   = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	U_flip   = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	S_inv    = <float*>malloc(n*sizeof(np.complex64_t))
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
cdef tuple _zresolvent(double[:,:] A, np.complex128_t f):

	cdef int ii, jj, n
	n = A.shape[0]

	cdef np.complex128_t *H_inv
	H_inv  = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	
	# COMPTE ELS SIGNES!
	for ii in range(n):
		for jj in range(n):
			if ii == jj:
				H_inv[ii*n + jj] = f - A[ii, jj]
			else:
				H_inv[ii*n + jj] = -A[ii, jj] 	

	cdef np.complex128_t *UT_flip
	cdef np.complex128_t *V_flip
	cdef np.complex128_t *U_flip
	cdef double *S_inv
	UT_flip  = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	V_flip   = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	U_flip   = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	S_inv    = <double*>malloc(n*sizeof(np.complex128_t))
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
def resolvent(real[:,:] X, object f):

	if real is double:
		return _zresolvent(X,np.complex128(f))
	else:
		return _cresolvent(X,np.complex64(f))


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _sflip_columns(float[:,:] A):
	'''
	
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
	'''
	
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
	'''
	
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
	'''
	
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

	'''
	if real_full is np.complex128_t:
		return _zflip_columns(A)
	elif real_full is np.complex64_t:
		return _cflip_columns(A)
	elif real_full is double:
		return _dflip_columns(A)
	else:
		return _sflip_columns(A)