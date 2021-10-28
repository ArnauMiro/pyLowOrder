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
from mpi4py  import MPI

from libc.stdlib   cimport malloc, free
from libc.string   cimport memcpy, memset
from mpi4py.libmpi cimport MPI_Comm
from mpi4py        cimport MPI

from ..utils.cr     import cr_start, cr_stop
from ..utils.errors import raiseError

cdef extern from "vector_matrix.h":
	cdef double c_vector_norm "vector_norm"(double *v, int start, int n)
	cdef void   c_matmul      "matmul"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_vecmat      "vecmat"(double *v, double *A, const int m, const int n)
cdef extern from "averaging.h":
	cdef void c_temporal_mean "temporal_mean"(double *out, double *X, const int m, const int n)
	cdef void c_subtract_mean "subtract_mean"(double *out, double *X, double *X_mean, const int m, const int n)
cdef extern from "svd.h":
	cdef int c_tsqr_svd "tsqr_svd"(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef int compute_truncation_residual(double *S, double res, const int n):
	'''
	Function which computes the accumulative residual of the vector S (of size n) and it
	returns truncation instant according to the desired residual, res, imposed by the user.
	'''
	cdef int ii
	cdef double accumulative
	cdef double normS = c_vector_norm(S,0,n)

	for ii in range(n):
		accumulative = c_vector_norm(S,ii,n)/normS
		if accumulative < res: return ii
	return n

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef void compute_truncation(double *Ur, double *Sr, double *VTr, double *U, 
	double *S, double *VT, const int m, const int n, const int N):
	'''
	U(m,n)   are the POD modes and must come preallocated.
	S(n)     are the singular values.
	VT(n,n)  are the right singular vectors (transposed).

	U, S and VT are copied to (they come preallocated):

	Ur(m,N)  are the POD modes and must come preallocated.
	Sr(N)    are the singular values.
	VTr(N,n) are the right singular vectors (transposed).
	'''
	cdef int ii, jj
	for jj in range(N):
		# Copy U into Ur
		for ii in range(m):
			Ur[N*ii+jj] = U[n*ii+jj]
		# Copy S into Sr
		Sr[jj] = S[jj]
		# Copy VT into VTr
		memcpy(VTr+n*jj,VT+n*jj,n*sizeof(double))


## POD run method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def run(double[:,:] X,int remove_mean=True):
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
	cr_start('POD.run',0)
	# Variables
	cdef int m = X.shape[0], n = X.shape[1], mn = min(m,n), retval
	cdef double *X_mean
	cdef double *Y
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	# Output arrays
	cdef np.ndarray[np.double_t,ndim=2] U = np.zeros((m,mn),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((mn,) ,dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] V = np.zeros((n,mn),dtype=np.double)
	# Allocate memory
	Y = <double*>malloc(m*n*sizeof(double))
	if remove_mean:
		X_mean = <double*>malloc(m*sizeof(double))
		# Compute temporal mean
		c_temporal_mean(X_mean,&X[0,0],m,n)
		# Compute substract temporal mean
		c_subtract_mean(Y,&X[0,0],X_mean,m,n)
		free(X_mean)
	else:
		memcpy(Y,&X[0,0],m*n*sizeof(double))
	# Compute SVD
	retval = c_tsqr_svd(&U[0,0],&S[0],&V[0,0],Y,m,n,MPI_COMM.ob_mpi)
	free(Y)
	# Return
	cr_stop('POD.run',0)
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V


## POD truncate method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def truncate(double[:,:] U, double[:] S, double[:,:] V, double r=1e-8):
	'''
	Truncate POD matrices (U,S,V) given a residual r.

	Inputs:
		- U(m,n)  are the POD modes.
		- S(n)    are the singular values.
		- V(n,n)  are the right singular vectors.
		- r       target residual (default 1e-8)

	Returns:
		- U(m,N)  are the POD modes (truncated at N).
		- S(N)    are the singular values (truncated at N).
		- V(N,n)  are the right singular vectors (truncated at N).
	'''
	cr_start('POD.truncate',0)
	cdef int m = U.shape[0], n = S.shape[0], N
	# Output arrays
	cdef np.ndarray[np.double_t,ndim=2] Ur, Vr
	cdef np.ndarray[np.double_t,ndim=1] Sr
	# Compute N using S
	N  = compute_truncation_residual(&S[0],r,n)
	# Allocate output arrays
	Ur = np.zeros((m,N),dtype=np.double)
	Sr = np.zeros((N,) ,dtype=np.double)
	Vr = np.zeros((N,n),dtype=np.double)
	# Truncate
	compute_truncation(&Ur[0,0],&Sr[0],&Vr[0,0],&U[0,0],&S[0],&V[0,0],m,n,N)
	# Return
	cr_stop('POD.truncate',0)
	return Ur, Sr, Vr


## POD reconstruct method
def reconstruct(double[:,:] U, double[:] S, double[:,:] V):
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
	cr_start('POD.reconstruct',0)
	cdef int m = U.shape[0], N = S.shape[0], n = V.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] X = np.zeros((m,n),dtype=np.double)
	cdef double *Vtmp
	# Copy V to Vtmp so V is not modified by the routine
	Vtmp = <double*>malloc(N*n*sizeof(double))
	memcpy(Vtmp,&V[0,0],N*n*sizeof(double))
	# Scale V by S doing V' = diag(S) x V
	c_vecmat(&S[0],Vtmp,N,n)
	# Compute X = U x V'
	c_matmul(&X[0,0],&U[0,0],Vtmp,m,n,N)
	# Return
	free(Vtmp)
	cr_stop('POD.reconstruct',0)
	return X