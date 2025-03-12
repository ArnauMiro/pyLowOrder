#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for POD.
#
# Last rev: 09/07/2021
from __future__ import print_function, division

cimport cython
cimport numpy as np

import numpy as np

from libc.stdlib     cimport malloc, free
from libc.string     cimport memcpy
from libc.time       cimport time
from ..vmmath.cfuncs cimport real
from ..vmmath.cfuncs cimport c_svector_norm, c_smatmul, c_svecmat, c_stemporal_mean, c_ssubtract_mean, c_stsqr_svd, c_srandomized_svd, c_scompute_truncation_residual, c_scompute_truncation
from ..vmmath.cfuncs cimport c_dvector_norm, c_dmatmul, c_dvecmat, c_dtemporal_mean, c_dsubtract_mean, c_dtsqr_svd, c_drandomized_svd, c_dcompute_truncation_residual, c_dcompute_truncation

from ..utils.cr       import cr, cr_start, cr_stop
from ..utils.errors   import raiseError


## POD run method
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _srun(float[:,:] X, int remove_mean, int randomized, int r, int q, int seed):
	'''
	Run POD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,n_temp_snapshots]: data matrix
		- remove_mean:                     whether or not to remove the mean flow

	Returns:
		- U:  are the POD modes.
		- S:  are the singular values.
		- V:  are the right singular vectors.
	'''
	# Variables
	cdef int m = X.shape[0], n = X.shape[1], mn = min(m,n), retval
	cdef float *X_mean
	cdef float *Y
	# Output arrays
	r = r if randomized else mn
	cdef np.ndarray[np.float32_t,ndim=2] U = np.zeros((m,r),dtype=np.float32) 
	cdef np.ndarray[np.float32_t,ndim=1] S = np.zeros((r,) ,dtype=np.float32) 
	cdef np.ndarray[np.float32_t,ndim=2] V = np.zeros((r,n),dtype=np.float32) 
	# Allocate memory
	Y = <float*>malloc(m*n*sizeof(float))
	if remove_mean:
		cr_start('POD.temporal_mean',0)
		X_mean = <float*>malloc(m*sizeof(float))
		# Compute temporal mean
		c_stemporal_mean(X_mean,&X[0,0],m,n)
		# Compute substract temporal mean
		c_ssubtract_mean(Y,&X[0,0],X_mean,m,n)
		free(X_mean)
		cr_stop('POD.temporal_mean',0)
	else:
		memcpy(Y,&X[0,0],m*n*sizeof(float))
	# Compute SVD
	cr_start('POD.SVD',0)
	if randomized:
		retval = c_srandomized_svd(&U[0,0],&S[0],&V[0,0],Y,m,n,r,q,seed)
	else:
		retval = c_stsqr_svd(&U[0,0],&S[0],&V[0,0],Y,m,n)
	cr_stop('POD.SVD',0)
	free(Y)
	# Return
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _drun(double[:,:] X, int remove_mean, int randomized, int r, int q, int seed):
	'''
	Run POD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,n_temp_snapshots]: data matrix
		- remove_mean:                     whether or not to remove the mean flow

	Returns:
		- U:  are the POD modes.
		- S:  are the singular values.
		- V:  are the right singular vectors.
	'''
	# Variables
	cdef int m = X.shape[0], n = X.shape[1], mn = min(m,n), retval
	cdef double *X_mean
	cdef double *Y
	# Output arrays
	r = r if randomized else mn
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,r),dtype=np.double) 
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((r,) ,dtype=np.double) 
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((r,n),dtype=np.double) 
	# Allocate memory
	Y = <double*>malloc(m*n*sizeof(double))
	if remove_mean:
		cr_start('POD.temporal_mean',0)
		X_mean = <double*>malloc(m*sizeof(double))
		# Compute temporal mean
		c_dtemporal_mean(X_mean,&X[0,0],m,n)
		# Compute substract temporal mean
		c_dsubtract_mean(Y,&X[0,0],X_mean,m,n)
		free(X_mean)
		cr_stop('POD.temporal_mean',0)
	else:
		memcpy(Y,&X[0,0],m*n*sizeof(double))
	# Compute SVD
	cr_start('POD.SVD',0)
	if randomized:
		retval = c_drandomized_svd(&U[0,0],&S[0],&V[0,0],Y,m,n,r,q,seed)
	else:
		retval = c_dtsqr_svd(&U[0,0],&S[0],&V[0,0],Y,m,n)
	cr_stop('POD.SVD',0)
	free(Y)
	# Return
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cr('POD.run')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def run(real[:,:] X, int remove_mean=True, int randomized=False, const int r=1, const int q=3, const int seed=-1):
	'''
	Run POD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,n_temp_snapshots]: data matrix
		- remove_mean:                     whether or not to remove the mean flow

	Returns:
		- U:  are the POD modes.
		- S:  are the singular values.
		- V:  are the right singular vectors.
	'''
	seed = <int>time(NULL) if seed < 0 else seed
	if real is double:
		return _drun(X,remove_mean, randomized, r, q, seed)
	else:
		return _srun(X,remove_mean, randomized, r, q, seed)


## POD truncate method
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _struncate(float[:,:] U, float[:] S, float[:,:] V, float r):
	'''
	Truncate POD matrices (U,S,V) given a residual r.

	Inputs:
		- U(m,n)  are the POD modes.
		- S(n)    are the singular values.
		- V(n,n)  are the right singular vectors.
		- r       target residual, number of modes, or cumulative energy threshold.
					* If r >= 1, it is treated as the number of modes.
					* If r < 1 and r > 0 it is treated as the residual target.
					* If r < 1 and r < 0 it is treated as the fraction of cumulative energy to retain.
					Note:  must be in (0,-1] and r = -1 is valid

	Returns:
		- U(m,N)  are the POD modes (truncated at N).
		- S(N)    are the singular values (truncated at N).
		- V(N,n)  are the right singular vectors (truncated at N).
	'''
	cdef int m = U.shape[0], n = V.shape[1], nmod = U.shape[1], N
	# Compute N using S
	N  = int(r) if r >=1 else c_scompute_truncation_residual(&S[0],r,n)
	# Allocate output arrays
	cdef np.ndarray[np.float32_t,ndim=2] Ur = np.zeros((m,N),dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] Sr = np.zeros((N,), dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] Vr = np.zeros((N,n),dtype=np.float32)
	# Truncate
	c_scompute_truncation(&Ur[0,0],&Sr[0],&Vr[0,0],&U[0,0],&S[0],&V[0,0],m,n,nmod,N)
	# Return
	return Ur, Sr, Vr

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dtruncate(double[:,:] U, double[:] S, double[:,:] V, double r):
	'''
	Truncate POD matrices (U,S,V) given a residual r.

	Inputs:
		- U(m,n)  are the POD modes.
		- S(n)    are the singular values.
		- V(n,n)  are the right singular vectors.
		- r       target residual, number of modes, or cumulative energy threshold.
					* If r >= 1, it is treated as the number of modes.
					* If r < 1 and r > 0 it is treated as the residual target.
					* If r < 1 and r < 0 it is treated as the fraction of cumulative energy to retain.
					Note:  must be in (0,-1] and r = -1 is valid

	Returns:
		- U(m,N)  are the POD modes (truncated at N).
		- S(N)    are the singular values (truncated at N).
		- V(N,n)  are the right singular vectors (truncated at N).
	'''
	cdef int m = U.shape[0], n = V.shape[1], N, nmod = U.shape[1]
	# Compute N using S
	N  = int(r) if r >=1 else c_dcompute_truncation_residual(&S[0],r,n)
	# Allocate output arrays
	cdef np.ndarray[np.double_t,ndim=2] Ur = np.zeros((m,N),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] Sr = np.zeros((N,), dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Vr = np.zeros((N,n),dtype=np.double)
	# Truncate
	c_dcompute_truncation(&Ur[0,0],&Sr[0],&Vr[0,0],&U[0,0],&S[0],&V[0,0],m,n,nmod,N)
	# Return
	return Ur, Sr, Vr

@cr('POD.truncate')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def truncate(real[:,:] U, real[:] S, real[:,:] V, real r=1e-8):
	'''
	Truncate POD matrices (U,S,V) given a residual r.

	Inputs:
		- U(m,n)  are the POD modes.
		- S(n)    are the singular values.
		- V(n,n)  are the right singular vectors.
		- r       target residual, number of modes, or cumulative energy threshold.
					* If r >= 1, it is treated as the number of modes.
					* If r < 1 and r > 0 it is treated as the residual target.
					* If r < 1 and r < 0 it is treated as the fraction of cumulative energy to retain.
					Note:  must be in (0,-1] and r = -1 is valid

	Returns:
		- U(m,N)  are the POD modes (truncated at N).
		- S(N)    are the singular values (truncated at N).
		- V(N,n)  are the right singular vectors (truncated at N).
	'''
	if real is double:
		return _dtruncate(U,S,V,r)
	else:
		return _struncate(U,S,V,r)


## POD reconstruct method
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _sreconstruct(float[:,:] U, float[:] S, float[:,:] V):
	'''
	Reconstruct the flow given the POD decomposition matrices
	that can be possibly truncated.
	N is the truncated size
	n is the number of snapshots

	Inputs:
		- U(m,N)  are the POD modes.
		- S(N)    are the singular values.
		- V(N,n)  are the right singular vectors.

	Outputs
		- X(m,n)  is the reconstructed flow.
	'''
	cdef int m = U.shape[0], N = S.shape[0], n = V.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] X = np.zeros((m,n),dtype=np.float32)
	cdef float *Vtmp
	# Copy V to Vtmp so V is not modified by the routine
	Vtmp = <float*>malloc(N*n*sizeof(float))
	memcpy(Vtmp,&V[0,0],N*n*sizeof(float))
	# Scale V by S doing V' = diag(S) x V
	c_svecmat(&S[0],Vtmp,N,n)
	# Compute X = U x V'
	c_smatmul(&X[0,0],&U[0,0],Vtmp,m,n,N)
	# Return
	free(Vtmp)
	return X

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dreconstruct(double[:,:] U, double[:] S, double[:,:] V):
	'''
	Reconstruct the flow given the POD decomposition matrices
	that can be possibly truncated.
	N is the truncated size
	n is the number of snapshots

	Inputs:
		- U(m,N)  are the POD modes.
		- S(N)    are the singular values.
		- V(N,n)  are the right singular vectors.

	Outputs
		- X(m,n)  is the reconstructed flow.
	'''
	cdef int m = U.shape[0], N = S.shape[0], n = V.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] X = np.zeros((m,n),dtype=np.double)
	cdef double *Vtmp
	# Copy V to Vtmp so V is not modified by the routine
	Vtmp = <double*>malloc(N*n*sizeof(double))
	memcpy(Vtmp,&V[0,0],N*n*sizeof(double))
	# Scale V by S doing V' = diag(S) x V
	c_dvecmat(&S[0],Vtmp,N,n)
	# Compute X = U x V'
	c_dmatmul(&X[0,0],&U[0,0],Vtmp,m,n,N)
	# Return
	free(Vtmp)
	return X

@cr('POD.reconstruct')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def reconstruct(real[:,:] U, real[:] S, real[:,:] V):
	'''
	Reconstruct the flow given the POD decomposition matrices
	that can be possibly truncated.
	N is the truncated size
	n is the number of snapshots

	Inputs:
		- U(m,N)  are the POD modes.
		- S(N)    are the singular values.
		- V(N,n)  are the right singular vectors.

	Outputs
		- X(m,n)  is the reconstructed flow.
	'''
	if real is double:
		return _dreconstruct(U,S,V)
	else:
		return _sreconstruct(U,S,V)
