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

from libc.stdlib   cimport malloc, free
from libc.string   cimport memcpy, memset
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

from ..utils.cr     import cr, cr_start, cr_stop
from ..utils.errors import raiseError

cdef extern from "vector_matrix.h":
	# Single precision
	cdef float  c_svector_norm "svector_norm"(float *v, int start, int n)
	cdef void   c_smatmul      "smatmul"(float *C, float *A, float *B, const int m, const int n, const int k)
	cdef void   c_svecmat      "svecmat"(float *v, float *A, const int m, const int n)
	# Double precision
	cdef double c_dvector_norm "dvector_norm"(double *v, int start, int n)
	cdef void   c_dmatmul      "dmatmul"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_dvecmat      "dvecmat"(double *v, double *A, const int m, const int n)
cdef extern from "averaging.h":
	# Single precision
	cdef void c_stemporal_mean "stemporal_mean"(float *out, float *X, const int m, const int n)
	cdef void c_ssubtract_mean "ssubtract_mean"(float *out, float *X, float *X_mean, const int m, const int n)
	# Double precision
	cdef void c_dtemporal_mean "dtemporal_mean"(double *out, double *X, const int m, const int n)
	cdef void c_dsubtract_mean "dsubtract_mean"(double *out, double *X, double *X_mean, const int m, const int n)
cdef extern from "svd.h":
	# Single precision
	cdef int c_stsqr_svd       "stsqr_svd"      (float *Ui, float *S, float *VT, float *Ai, const int m, const int n, MPI_Comm comm)
	cdef int c_srandomized_svd "srandomized_svd"(float *Ui, float *S, float *VT, float *Ai,   const int m, const int n, const int r, const int q, unsigned int seed, MPI_Comm comm)
	# Double precision
	cdef int c_dtsqr_svd       "dtsqr_svd"      (double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm)
	cdef int c_drandomized_svd "drandomized_svd"(double *Ui, double *S, double *VT, double *Ai,  const int m, const int n, const int r, const int q, unsigned int seed, MPI_Comm comm)
cdef extern from "truncation.h":
	# Single precision
	cdef int  c_scompute_truncation_residual "scompute_truncation_residual"(float *S, float res, const int n)
	cdef void c_scompute_truncation          "scompute_truncation"(float *Ur, float *Sr, float *VTr, float *U, float *S, float *VT, const int m, const int n, const int nmod, const int N)
	# Double precision
	cdef int  c_dcompute_truncation_residual "dcompute_truncation_residual"(double *S, double res, const int n)
	cdef void c_dcompute_truncation          "dcompute_truncation"(double *Ur, double *Sr, double *VTr, double *U, double *S, double *VT, const int m, const int n, const int nmod, const int N)


## Fused type between double and complex
ctypedef fused real:
	float
	double


## POD run method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _srun(float[:,:] X, int remove_mean, int randomized, int r, int q):
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
	cdef unsigned int seed = <int>time(NULL)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
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
		retval = c_srandomized_svd(&U[0,0],&S[0],&V[0,0],Y,m,n,r,q,seed,MPI_COMM.ob_mpi)
	else:
		retval = c_stsqr_svd(&U[0,0],&S[0],&V[0,0],Y,m,n,MPI_COMM.ob_mpi)
	cr_stop('POD.SVD',0)
	free(Y)
	# Return
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _drun(double[:,:] X, int remove_mean, int randomized, int r, int q):
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
	cdef unsigned int seed = <int>time(NULL)
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
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
		retval = c_drandomized_svd(&U[0,0],&S[0],&V[0,0],Y,m,n,r,q,seed,MPI_COMM.ob_mpi)
	else:
		retval = c_dtsqr_svd(&U[0,0],&S[0],&V[0,0],Y,m,n,MPI_COMM.ob_mpi)
	cr_stop('POD.SVD',0)
	free(Y)
	# Return
	if not retval == 0: raiseError('Problems computing SVD!')
	return U,S,V

@cr('POD.run')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def run(real[:,:] X, int remove_mean=True, int randomized=False, int r=1, int q=3):
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
	if real is double:
		return _drun(X,remove_mean, randomized, r, q)
	else:
		return _srun(X,remove_mean, randomized, r, q)


## POD truncate method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _struncate(float[:,:] U, float[:] S, float[:,:] V, float r):
	'''
	Truncate POD matrices (U,S,V) given a residual r.

	Inputs:
		- U(m,nmod)  are the POD modes.
		- S(nmod)    are the singular values.
		- V(nmod,n)  are the right singular vectors.
		- r       target residual (default 1e-8)
		If the SVD was done with the randomized algorithm, nmod < n but should always be larger than the target number of modes after truncation N

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

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dtruncate(double[:,:] U, double[:] S, double[:,:] V, double r):
	'''
	Truncate POD matrices (U,S,V) given a residual r.

	Inputs:
		- U(m,nmod)  are the POD modes.
		- S(nmod)    are the singular values.
		- V(nmod,n)  are the right singular vectors.
		- r       target residual (default 1e-8)
		If the SVD was done with the randomized algorithm, nmod < n but should always be larger than the target number of modes after truncation N


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
		- r       target residual (default 1e-8)

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
