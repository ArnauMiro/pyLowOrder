#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for RES.
#
# Last rev: 30/04/2026
from __future__ import print_function, division

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
from libc.stdlib     cimport malloc, calloc, free
from libc.string     cimport memcpy, memset
from libc.math       cimport sqrt, log, atan2
from ..vmmath.cfuncs cimport real, real_complex, real_float, real_double
from ..vmmath.cfuncs cimport c_svecmat, c_sconcatenate, c_sseparate, c_stsqr_svd, c_sremove_rows, c_scompute_truncation_residual, c_scompute_truncation, c_stranspose, c_smatmul, c_smatmulp
from ..vmmath.cfuncs cimport c_dvecmat, c_dconcatenate, c_dseparate, c_dtsqr_svd, c_dremove_rows, c_dcompute_truncation_residual, c_dcompute_truncation, c_dtranspose, c_dmatmul, c_dmatmulp
from ..vmmath.cfuncs cimport c_csvd, c_cdagger, c_cmatmul, c_cmatmulp, c_cvecmat, c_ccholesky, c_cinverse, c_cresolvent
from ..vmmath.cfuncs cimport c_zsvd, c_zdagger, c_zmatmul, c_zmatmulp, c_zvecmat, c_zcholesky, c_zinverse, c_zresolvent
from ..vmmath.linear cimport _sconcatenate, _slinear_operator, _sseparate, _cresolvent
from ..vmmath.linear cimport _dconcatenate, _dlinear_operator, _dseparate, _zresolvent

from ..utils.cr       import cr, cr_start, cr_stop
from ..utils.errors   import raiseError


## RES run method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _crun(np.complex64_t[:,:] Phi, float[:] delta, float[:] omega, float f, float[:] Q=None):
	'''
    Resolvent Analysis of snapshot matrix X
    Inputs:
        - X[ndims*nmesh,n_temp_snapshots]: data matrix
        - delta: damping ratio of each mode
        - omega: frequency of each mode
        - f: target frequency
        - Q: weighting matrix
    Returns:
        - U_res: response modes
        - S: emergy gains
        - V_res: forcing modes
    '''
	# Variables
	cdef int m = Phi.shape[0], n = Phi.shape[1]
	cdef int ii, retval

	# Compute the resolvent operator
	cdef np.complex64_t *Omega
	cdef np.complex64_t *H
	Omega  = <np.complex64_t*>malloc(n*sizeof(np.complex64_t))
	H  = <np.complex64_t*>malloc(n*sizeof(np.complex64_t))
	cr_start('RES.resolvent_operator', 0)
	for ii in range(n):
		Omega[ii] = delta[ii] + J * omega[ii]
		H[ii] = 1 / (-J * f - Omega[ii])
	free(Omega)
	cr_stop('RES.resolvent_operator', 0)

	# Compute the Qhat (named Fhat_dagger for convenience)
	cr_start('RES.Qhat', 0)
	cdef np.complex64_t *Phi_dagger
	cdef np.complex64_t *Phi_aux
	cdef np.complex64_t *Fhat_dagger
	Phi_dagger = <np.complex64_t*>malloc(n*m*sizeof(np.complex64_t))
	Phi_aux = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
	Fhat_dagger = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	cdef np.ndarray[np.complex64_t, ndim=1] Q_aux = np.zeros((m),dtype=np.complex64)
	c_cdagger(&Phi[0,0], Phi_dagger, m, n)
	if Q is None:
		c_cmatmulp(Fhat_dagger, Phi_dagger, &Phi[0,0], n, n, m)
	else:
		memcpy(Phi_aux, &Phi[0,0], m*n*sizeof(np.complex64_t))
		Q_aux = np.array(Q, dtype=np.complex64)
		c_cvecmat(&Q_aux[0], Phi_aux, m, n) # Phi_aux is overwritten
		c_cmatmulp(Fhat_dagger, Phi_dagger, Phi_aux, n, n, m)
	free(Phi_aux)
	free(Phi_dagger)
	cr_stop('RES.Qhat', 0)

	# Compute the Choleski decomposition
	cr_start('RES.Choleski', 0)
	cdef np.complex64_t *Fhat
	Fhat = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	retval = c_ccholesky(Fhat_dagger, n)
	if not retval == 0: raiseError('Problems computing Cholesky factorization!')
	c_cdagger(Fhat_dagger, Fhat, n, n)
	cdef np.complex64_t *Fhat_inv
	Fhat_inv = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	memcpy(Fhat_inv, Fhat, n*n*sizeof(np.complex64_t))
	retval = c_cinverse(Fhat_inv, n, 'U')
	if not retval == 0: raiseError('Problems computing the Inverse!')
	free(Fhat_dagger)
	cr_stop('RES.Choleski', 0)

	# Compute Hhat
	cr_start('RES.Hhat', 0)
	cdef np.complex64_t *Hhat
	cdef np.complex64_t *Fhat_aux
	Hhat = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	Fhat_aux = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	memcpy(Fhat_aux, Fhat_inv, n*n*sizeof(np.complex64_t))
	c_cvecmat(H, Fhat_aux, n, n)
	c_cmatmul(Hhat, Fhat, Fhat_aux, n, n, n)
	free(H)
	free(Fhat)
	free(Fhat_aux)
	cr_stop('RES.Hhat', 0)

	# Compute the svd
	cr_start('RES.svd', 0)
	cdef np.complex64_t *U
	cdef np.ndarray[np.float32_t,ndim=1] S = np.zeros((n),dtype=np.float32)
	cdef np.complex64_t *V
	cdef np.complex64_t *Vt
	U = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	V = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	Vt = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	c_csvd(U, &S[0], Vt, Hhat, n, n)
	c_cdagger(Vt, V, n, n)
	free(Hhat)
	free(Vt)
	cr_stop('RES.svd', 0)

	# Compute the projection
	cr_start('RES.projection', 0)
	cdef np.complex64_t *U_aux
	cdef np.complex64_t *V_aux
	U_aux = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
	V_aux = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
	cdef np.ndarray[np.complex64_t,ndim=2] U_res = np.zeros((m,n),dtype=np.complex64)
	cdef np.ndarray[np.complex64_t,ndim=2] V_res = np.zeros((m,n),dtype=np.complex64)
	c_cmatmul(U_aux, Fhat_inv, U, n, n, n)
	c_cmatmul(V_aux, Fhat_inv, V, n, n, n)
	c_cmatmul(&U_res[0,0], &Phi[0,0], U_aux, m, n, n)
	c_cmatmul(&V_res[0,0], &Phi[0,0], V_aux, m, n, n)
	free(Fhat_inv)
	free(U)
	free(U_aux)
	free(V)
	free(V_aux)
	cr_stop('RES.projection', 0)

	return U_res, S, V_res

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _zrun(np.complex128_t[:,:] Phi, double[:] delta, double[:] omega, double f, double[:] Q=None):
	'''
    Resolvent Analysis of snapshot matrix X
    Inputs:
        - X[ndims*nmesh,n_temp_snapshots]: data matrix
        - delta: damping ratio of each mode
        - omega: frequency of each mode
        - f: target frequency
        - Q: weighting matrix
    Returns:
        - U_res: response modes
        - S: emergy gains
        - V_res: forcing modes
    '''
	# Variables
	cdef int m = Phi.shape[0], n = Phi.shape[1]
	cdef int ii, retval

	# Compute the resolvent operator
	cdef np.complex128_t *Omega
	cdef np.complex128_t *H
	Omega  = <np.complex128_t*>malloc(n*sizeof(np.complex128_t))
	H  = <np.complex128_t*>malloc(n*sizeof(np.complex128_t))
	cr_start('RES.resolvent_operator', 0)
	for ii in range(n):
		Omega[ii] = delta[ii] + J * omega[ii]
		H[ii] = 1 / (-J * f - Omega[ii])
	free(Omega)
	cr_stop('RES.resolvent_operator', 0)

	# Compute the Qhat (named Fhat for convenience)
	cr_start('RES.Qhat', 0)
	cdef np.complex128_t *Phi_dagger
	cdef np.complex128_t *Phi_aux
	cdef np.complex128_t *Fhat_dagger
	Phi_dagger = <np.complex128_t*>malloc(n*m*sizeof(np.complex128_t))
	Phi_aux = <np.complex128_t*>malloc(m*n*sizeof(np.complex128_t))
	Fhat_dagger = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	cdef np.ndarray[np.complex128_t,ndim=1] Q_aux = np.zeros((m),dtype=np.complex128)
	c_zdagger(&Phi[0,0], Phi_dagger, m, n)
	if Q is None:
		c_zmatmulp(Fhat_dagger, Phi_dagger, &Phi[0,0], n, n, m)
	else:
		memcpy(Phi_aux, &Phi[0,0], m*n*sizeof(np.complex128_t))
		Q_aux = np.array(Q, dtype=np.complex128)
		c_zvecmat(&Q_aux[0], Phi_aux, m, n) # Phi_aux is overwritten
		c_zmatmulp(Fhat_dagger, Phi_dagger, Phi_aux, n, n, m)
	free(Phi_aux)
	free(Phi_dagger)
	cr_stop('RES.Qhat', 0)

	# Compute the Choleski decomposition
	cr_start('RES.Choleski', 0)
	cdef np.complex128_t *Fhat
	Fhat = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	retval = c_zcholesky(Fhat_dagger, n)
	if not retval == 0: raiseError('Problems computing Cholesky factorization!')
	c_zdagger(Fhat_dagger, Fhat, n, n)
	cdef np.complex128_t *Fhat_inv
	Fhat_inv = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	memcpy(Fhat_inv, Fhat, n*n*sizeof(np.complex128_t))
	retval = c_zinverse(Fhat_inv, n, 'U')
	if not retval == 0: raiseError('Problems computing the Inverse!')
	free(Fhat_dagger)
	cr_stop('RES.Choleski', 0)

	# Compute Hhat
	cr_start('RES.Hhat', 0)
	cdef np.complex128_t *Hhat
	cdef np.complex128_t *Fhat_aux
	Hhat = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	Fhat_aux = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	memcpy(Fhat_aux, Fhat_inv, n*n*sizeof(np.complex128_t))
	c_zvecmat(H, Fhat_aux, n, n)
	c_zmatmul(Hhat, Fhat, Fhat_aux, n, n, n)
	free(H)
	free(Fhat)
	free(Fhat_aux)
	cr_stop('RES.Hhat', 0)

	# Compute the svd
	cr_start('RES.svd', 0)
	cdef np.complex128_t *U
	cdef np.ndarray[np.float64_t,ndim=1] S = np.zeros((n),dtype=np.float64)
	cdef np.complex128_t *V
	cdef np.complex128_t *Vt
	U = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	V = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	Vt = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	c_zsvd(U, &S[0], Vt, Hhat, n, n)
	c_zdagger(Vt, V, n, n)
	free(Hhat)
	free(Vt)
	cr_stop('RES.svd', 0)

	# Compute the projection
	cr_start('RES.projection', 0)
	cdef np.complex128_t *U_aux
	cdef np.complex128_t *V_aux
	U_aux = <np.complex128_t*>malloc(m*n*sizeof(np.complex128_t))
	V_aux = <np.complex128_t*>malloc(m*n*sizeof(np.complex128_t))
	cdef np.ndarray[np.complex128_t,ndim=2] U_res = np.zeros((m,n),dtype=np.complex128)
	cdef np.ndarray[np.complex128_t,ndim=2] V_res = np.zeros((m,n),dtype=np.complex128)
	c_zmatmul(U_aux, Fhat_inv, U, n, n, n)
	c_zmatmul(V_aux, Fhat_inv, V, n, n, n)
	c_zmatmul(&U_res[0,0], &Phi[0,0], U_aux, m, n, n)
	c_zmatmul(&V_res[0,0], &Phi[0,0], V_aux, m, n, n)
	free(Fhat_inv)
	free(U)
	free(U_aux)
	free(V)
	free(V_aux)
	cr_stop('RES.projection', 0)

	return U_res, S, V_res

@cr('RES.run')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def run(real_complex[:,:] Phi, real[:] delta, real[:] omega, real f, real[:] Q=None):
	'''
    Resolvent Analysis of snapshot matrix X
    Inputs:
        - X[ndims*nmesh,n_temp_snapshots]: data matrix
        - delta: damping ratio of each mode
        - omega: frequency of each mode
        - f: target frequency
        - Q: weighting matrix
    Returns:
        - U_res: response modes
        - S: emergy gains
        - V_res: forcing modes
    '''
	if real_complex is np.complex128_t:
		return _zrun(Phi, delta, omega, f, Q)
	else:
		return _crun(Phi, delta, omega, f, Q)

def _crun_old(float[:,:] X, np.complex64_t w, float r, int remove_mean, float[:] Q):

	# Variables
	cdef int m, n, ii, m_aux
	m = X.shape[0]
	n = X.shape[1]

	if m > (n - 1):
		m_aux = m
	else:
		m_aux = n-1

	cdef float *Y
	cdef float *Z
	Y = <float*>calloc(m_aux*(n-1), sizeof(float))
	Z = <float*>malloc(m*(n-1)*sizeof(float))
	if Q is not None:
		c_svecmat(&Q[0], &X[0,0], m, n)

	# Create the snapshot of matrices and separate it into Y, Z
	c_sseparate(Y, Z, &X[0,0], m, n, remove_mean)

	# Compute the linear operator in lower dimension
	cdef int icol, irow, retval
	
	## Compute SVD
	cdef float *U_aux
	cdef float *S_all
	cdef float *VT_all
	U_aux  = <float*>malloc(m_aux*(n-1)*sizeof(float))
	S_all  = <float*>malloc((n-1)*sizeof(float))
	VT_all = <float*>malloc((n-1)*(n-1)*sizeof(float))

	retval = c_stsqr_svd(U_aux, S_all, VT_all, Y, m_aux, (n-1))
	free(Y)

	cdef float *U_all
	U_all = <float*>malloc(m_aux*(n-1)*sizeof(float))
	c_sremove_rows(U_aux, U_all, m, (n-1))
	free(U_aux)

	## Truncate
	cdef int nr
	if r > 1:
		nr = int(r)
	else:
		nr = c_scompute_truncation_residual(S_all, r, (n-1))

	cdef float *Ur
	cdef float *Sr
	cdef float *VTr
	Ur  = <float*>malloc(m*nr*sizeof(float))
	Sr  = <float*>malloc(nr*sizeof(float))
	VTr = <float*>malloc(nr*(n-1)*sizeof(float))

	c_scompute_truncation(Ur, Sr, VTr, U_all, S_all, VT_all, m, (n-1), (n-1), nr)
	free(U_all)
	free(S_all)
	free(VT_all)

	# Project Jacobian of the snapshots into the POD basis
	cdef float *aux1
	cdef float *aux2
	cdef float *aux3
	cdef float *Atilde
	cdef float *Urt
	aux1   = <float*>malloc(nr*(n-1)*sizeof(float))
	aux2   = <float*>malloc(nr*(n-1)*sizeof(float))
	aux3   = <float*>malloc(nr*sizeof(float))
	Atilde = <float*>malloc(nr*nr*sizeof(float))
	Urt    = <float*>malloc(nr*m*sizeof(float))
	c_stranspose(Ur, Urt, m, nr)
	c_smatmulp(aux1, Urt, Z, nr, n-1, m)
	free(Z)

	for icol in range(n-1):
		for irow in range(nr):
			aux2[icol*nr + irow] = VTr[irow*(n-1) + icol]/Sr[irow]
	c_smatmul(Atilde, aux1, aux2, nr, nr, n-1)
	free(aux1)
	free(aux2)
	free(aux3)
	free(Urt)
	free(VTr)
	free(Sr)

	cdef np.ndarray[np.float32_t,ndim=1] S = np.zeros((nr),dtype=np.float32)
	cdef np.complex64_t *U2
	cdef np.complex64_t *V2
	U2 = <np.complex64_t*>malloc(nr*nr*sizeof(np.complex64_t))
	V2 = <np.complex64_t*>malloc(nr*nr*sizeof(np.complex64_t))

	c_cresolvent(U2, &S[0], V2, Atilde, w, nr)
	free(Atilde)

	cdef np.complex64_t *U_c
	U_c = <np.complex64_t*>malloc(m*nr*sizeof(np.complex64_t))

	for ii in range(m*nr):
		U_c[ii] = Ur[ii]
	free(Ur)
	
	cdef np.ndarray[np.complex64_t,ndim=2] U = np.zeros((m,nr),dtype=np.complex64)
	cdef np.ndarray[np.complex64_t,ndim=2] V = np.zeros((m,nr),dtype=np.complex64)
	c_cmatmul(&U[0,0], U_c, U2, m, nr, nr)
	c_cmatmul(&V[0,0], U_c, V2, m, nr, nr)
	free(U2)
	free(V2)
	free(U_c)

	cdef np.complex64_t *Q_inv
	Q_inv = <np.complex64_t*>malloc(m*sizeof(np.complex64_t))
	if Q is not None:
		for ii in range(m):
			Q_inv[ii] = 1.0 / Q[ii]
		c_cvecmat(Q_inv, &U[0,0], m, nr)
		c_cvecmat(Q_inv, &V[0,0], m, nr)
	free(Q_inv)

	return U, S, V

def _zrun_old(double[:,:] X, np.complex128_t w, double r, int remove_mean, double[:] Q):

	# Variables
	cdef int m, n, ii, m_aux
	m = X.shape[0]
	n = X.shape[1]

	if m > (n - 1):
		m_aux = m
	else:
		m_aux = n-1

	cdef double *Y
	cdef double *Z
	Y = <double*>calloc(m_aux*(n-1), sizeof(double))
	Z = <double*>malloc(m*(n-1)*sizeof(double))
	if Q is not None:
		c_dvecmat(&Q[0], &X[0,0], m, n)

	# Create the snapshot of matrices and separate it into Y, Z
	c_dseparate(Y, Z, &X[0,0], m, n, remove_mean)

	# Compute the linear operator in lower dimension
	cdef int icol, irow, retval
	
	## Compute SVD
	cdef double *U_aux
	cdef double *S_all
	cdef double *VT_all
	U_aux  = <double*>malloc(m_aux*(n-1)*sizeof(double))
	S_all  = <double*>malloc((n-1)*sizeof(double))
	VT_all = <double*>malloc((n-1)*(n-1)*sizeof(double))

	retval = c_dtsqr_svd(U_aux, S_all, VT_all, Y, m_aux, (n-1))
	free(Y)

	cdef double *U_all
	U_all = <double*>malloc(m_aux*(n-1)*sizeof(double))
	c_dremove_rows(U_aux, U_all, m, (n-1))
	free(U_aux)

	## Truncate
	cdef int nr
	if r > 1:
		nr = int(r)
	else:
		nr = c_dcompute_truncation_residual(S_all, r, (n-1))

	cdef double *Ur
	cdef double *Sr
	cdef double *VTr
	Ur  = <double*>malloc(m*nr*sizeof(double))
	Sr  = <double*>malloc(nr*sizeof(double))
	VTr = <double*>malloc(nr*(n-1)*sizeof(double))

	c_dcompute_truncation(Ur, Sr, VTr, U_all, S_all, VT_all, m, (n-1), (n-1), nr)
	free(U_all)
	free(S_all)
	free(VT_all)

	# Project Jacobian of the snapshots into the POD basis
	cdef double *aux1
	cdef double *aux2
	cdef double *aux3
	cdef double *Atilde
	cdef double *Urt
	aux1   = <double*>malloc(nr*(n-1)*sizeof(double))
	aux2   = <double*>malloc(nr*(n-1)*sizeof(double))
	aux3   = <double*>malloc(nr*sizeof(double))
	Atilde = <double*>malloc(nr*nr*sizeof(double))
	Urt    = <double*>malloc(nr*m*sizeof(double))
	c_dtranspose(Ur, Urt, m, nr)
	c_dmatmulp(aux1, Urt, Z, nr, n-1, m)
	free(Z)

	for icol in range(n-1):
		for irow in range(nr):
			aux2[icol*nr + irow] = VTr[irow*(n-1) + icol]/Sr[irow]
	c_dmatmul(Atilde, aux1, aux2, nr, nr, n-1)
	free(aux1)
	free(aux2)
	free(aux3)
	free(Urt)
	free(VTr)
	free(Sr)

	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((nr),dtype=np.double)
	cdef np.complex128_t *U2
	cdef np.complex128_t *V2
	U2 = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))
	V2 = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))

	c_zresolvent(U2, &S[0], V2, Atilde, w, nr)
	free(Atilde)

	cdef np.complex128_t *U_c
	U_c = <np.complex128_t*>malloc(m*nr*sizeof(np.complex128_t))

	for ii in range(m*nr):
		U_c[ii] = Ur[ii]
	free(Ur)
	
	cdef np.ndarray[np.complex128_t,ndim=2] U = np.zeros((m,nr),dtype=np.complex128)
	cdef np.ndarray[np.complex128_t,ndim=2] V = np.zeros((m,nr),dtype=np.complex128)
	c_zmatmul(&U[0,0], U_c, U2, m, nr, nr)
	c_zmatmul(&V[0,0], U_c, V2, m, nr, nr)
	free(U2)
	free(V2)
	free(U_c)

	cdef np.complex128_t *Q_inv
	Q_inv = <np.complex128_t*>malloc(m*sizeof(np.complex128_t))
	if Q is not None:
		for ii in range(m):
			Q_inv[ii] = 1.0 / Q[ii]
		c_zvecmat(Q_inv, &U[0,0], m, nr)
		c_zvecmat(Q_inv, &V[0,0], m, nr)
	free(Q_inv)

	return U, S, V

def _crun_new(list X, np.complex64_t w, float r, int remove_mean, float[:] Q):

	# Variables
	cdef int m, ii, m_aux, n_total, n_matrix, n_local
	m = X[0].shape[0]
	n_matrix = len(X)

	cdef float **X_list
	cdef int *n_list
	X_pointer = <float**>malloc(n_matrix*sizeof(float*))
	n_list = <int*>malloc(n_matrix*sizeof(int))
	### LOOP X
	n_total = 0
	cdef float[:, ::1] M 
	for ii in range(len(X)):
		M = X[ii]
		n_local = M.shape[1]
		n_total += n_local - 1
		if Q is not None:
			c_svecmat(&Q[0], &M[0,0], m, n_local)
		n_list[ii] = n_local
		X_pointer[ii] = &M[0,0]

	if m > n_total:
		m_aux = m
	else:
		m_aux = n_total

	cdef float *Y
	cdef float *Z
	Y = <float*>calloc(m_aux*n_total, sizeof(float))
	Z = <float*>malloc(m*n_total*sizeof(float))

	# Create the snapshot of matrices and separate it into Y, Z
	c_sconcatenate(Y, Z, X_pointer, n_list, n_matrix, m, n_total, remove_mean)
	### MODIFICAR A SOBRE D'AIXO!!!
	# Compute the linear operator in lower dimension
	cdef int icol, irow, retval
	
	## Compute SVD
	cdef float *U_aux
	cdef float *S_all
	cdef float *VT_all
	U_aux  = <float*>malloc(m_aux*n_total*sizeof(float))
	S_all  = <float*>malloc(n_total*sizeof(float))
	VT_all = <float*>malloc(n_total*n_total*sizeof(float))

	retval = c_stsqr_svd(U_aux, S_all, VT_all, Y, m_aux, n_total)
	free(Y)

	cdef float *U_all
	U_all = <float*>malloc(m_aux*n_total*sizeof(float))
	c_sremove_rows(U_aux, U_all, m, n_total)
	free(U_aux)

	## Truncate
	cdef int nr
	if r > 1:
		nr = int(r)
	else:
		nr = c_scompute_truncation_residual(S_all, r, n_total)

	cdef float *Ur
	cdef float *Sr
	cdef float *VTr
	Ur  = <float*>malloc(m*nr*sizeof(float))
	Sr  = <float*>malloc(nr*sizeof(float))
	VTr = <float*>malloc(nr*n_total*sizeof(float))

	c_scompute_truncation(Ur, Sr, VTr, U_all, S_all, VT_all, m, n_total, n_total, nr)
	free(U_all)
	free(S_all)
	free(VT_all)

	# Project Jacobian of the snapshots into the POD basis
	cdef float *aux1
	cdef float *aux2
	cdef float *aux3
	cdef float *Atilde
	cdef float *Urt
	aux1   = <float*>malloc(nr*n_total*sizeof(float))
	aux2   = <float*>malloc(nr*n_total*sizeof(float))
	aux3   = <float*>malloc(nr*sizeof(float))
	Atilde = <float*>malloc(nr*nr*sizeof(float))
	Urt    = <float*>malloc(nr*m*sizeof(float))
	c_stranspose(Ur, Urt, m, nr)
	c_smatmulp(aux1, Urt, Z, nr, n_total, m)
	free(Z)

	for icol in range(n_total):
		for irow in range(nr):
			aux2[icol*nr + irow] = VTr[irow*n_total + icol]/Sr[irow]
	c_smatmul(Atilde, aux1, aux2, nr, nr, n_total)
	free(aux1)
	free(aux2)
	free(aux3)
	free(Urt)
	free(VTr)
	free(Sr)

	cdef np.ndarray[np.float32_t,ndim=1] S = np.zeros((nr),dtype=np.float32)
	cdef np.complex64_t *U2
	cdef np.complex64_t *V2
	U2 = <np.complex64_t*>malloc(nr*nr*sizeof(np.complex64_t))
	V2 = <np.complex64_t*>malloc(nr*nr*sizeof(np.complex64_t))

	c_cresolvent(U2, &S[0], V2, Atilde, w, nr)
	free(Atilde)

	cdef np.complex64_t *U_c
	U_c = <np.complex64_t*>malloc(m*nr*sizeof(np.complex64_t))

	for ii in range(m*nr):
		U_c[ii] = Ur[ii]
	free(Ur)
	
	cdef np.ndarray[np.complex64_t,ndim=2] U = np.zeros((m,nr),dtype=np.complex64)
	cdef np.ndarray[np.complex64_t,ndim=2] V = np.zeros((m,nr),dtype=np.complex64)
	c_cmatmul(&U[0,0], U_c, U2, m, nr, nr)
	c_cmatmul(&V[0,0], U_c, V2, m, nr, nr)
	free(U2)
	free(V2)
	free(U_c)

	cdef np.complex64_t *Q_inv
	Q_inv = <np.complex64_t*>malloc(m*sizeof(np.complex64_t))
	if Q is not None:
		for ii in range(m):
			Q_inv[ii] = 1.0 / Q[ii]
		c_cvecmat(Q_inv, &U[0,0], m, nr)
		c_cvecmat(Q_inv, &V[0,0], m, nr)
	free(Q_inv)

	return U, S, V

def _zrun_new(list X, np.complex128_t w, double r, int remove_mean, double[:] Q):

	# Variables
	cdef int m, ii, m_aux, n_total, n_matrix, n_local
	m = X[0].shape[0]
	n_matrix = len(X)

	cdef double **X_list
	cdef int *n_list
	X_pointer = <double**>malloc(n_matrix*sizeof(double*))
	n_list = <int*>malloc(n_matrix*sizeof(int))
	### LOOP X
	n_total = 0
	cdef double[:, ::1] M 
	for ii in range(len(X)):
		M = X[ii]
		n_local = M.shape[1]
		n_total += n_local - 1
		if Q is not None:
			c_dvecmat(&Q[0], &M[0,0], m, n_local)
		n_list[ii] = n_local
		X_pointer[ii] = &M[0,0]

	if m > n_total:
		m_aux = m
	else:
		m_aux = n_total

	cdef double *Y
	cdef double *Z
	Y = <double*>calloc(m_aux*n_total, sizeof(double))
	Z = <double*>malloc(m*n_total*sizeof(double))

	# Create the snapshot of matrices and separate it into Y, Z
	c_dconcatenate(Y, Z, X_pointer, n_list, n_matrix, m, n_total, remove_mean)
	### MODIFICAR A SOBRE D'AIXO!!!
	# Compute the linear operator in lower dimension
	cdef int icol, irow, retval
	
	## Compute SVD
	cdef double *U_aux
	cdef double *S_all
	cdef double *VT_all
	U_aux  = <double*>malloc(m_aux*n_total*sizeof(double))
	S_all  = <double*>malloc(n_total*sizeof(double))
	VT_all = <double*>malloc(n_total*n_total*sizeof(double))

	retval = c_dtsqr_svd(U_aux, S_all, VT_all, Y, m_aux, n_total)
	free(Y)

	cdef double *U_all
	U_all = <double*>malloc(m_aux*n_total*sizeof(double))
	c_dremove_rows(U_aux, U_all, m, n_total)
	free(U_aux)

	## Truncate
	cdef int nr
	if r > 1:
		nr = int(r)
	else:
		nr = c_dcompute_truncation_residual(S_all, r, n_total)

	cdef double *Ur
	cdef double *Sr
	cdef double *VTr
	Ur  = <double*>malloc(m*nr*sizeof(double))
	Sr  = <double*>malloc(nr*sizeof(double))
	VTr = <double*>malloc(nr*n_total*sizeof(double))

	c_dcompute_truncation(Ur, Sr, VTr, U_all, S_all, VT_all, m, n_total, n_total, nr)
	free(U_all)
	free(S_all)
	free(VT_all)

	# Project Jacobian of the snapshots into the POD basis
	cdef double *aux1
	cdef double *aux2
	cdef double *aux3
	cdef double *Atilde
	cdef double *Urt
	aux1   = <double*>malloc(nr*n_total*sizeof(double))
	aux2   = <double*>malloc(nr*n_total*sizeof(double))
	aux3   = <double*>malloc(nr*sizeof(double))
	Atilde = <double*>malloc(nr*nr*sizeof(double))
	Urt    = <double*>malloc(nr*m*sizeof(double))
	c_dtranspose(Ur, Urt, m, nr)
	c_dmatmulp(aux1, Urt, Z, nr, n_total, m)
	free(Z)

	for icol in range(n_total):
		for irow in range(nr):
			aux2[icol*nr + irow] = VTr[irow*n_total + icol]/Sr[irow]
	c_dmatmul(Atilde, aux1, aux2, nr, nr, n_total)
	free(aux1)
	free(aux2)
	free(aux3)
	free(Urt)
	free(VTr)
	free(Sr)

	cdef np.ndarray[np.double_t,ndim=1] S = np.zeros((nr),dtype=np.double)
	cdef np.complex128_t *U2
	cdef np.complex128_t *V2
	U2 = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))
	V2 = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))

	c_zresolvent(U2, &S[0], V2, Atilde, w, nr)
	free(Atilde)

	cdef np.complex128_t *U_c
	U_c = <np.complex128_t*>malloc(m*nr*sizeof(np.complex128_t))

	for ii in range(m*nr):
		U_c[ii] = Ur[ii]
	free(Ur)
	
	cdef np.ndarray[np.complex128_t,ndim=2] U = np.zeros((m,nr),dtype=np.complex128)
	cdef np.ndarray[np.complex128_t,ndim=2] V = np.zeros((m,nr),dtype=np.complex128)
	c_zmatmul(&U[0,0], U_c, U2, m, nr, nr)
	c_zmatmul(&V[0,0], U_c, V2, m, nr, nr)
	free(U2)
	free(V2)
	free(U_c)

	cdef np.complex128_t *Q_inv
	Q_inv = <np.complex128_t*>malloc(m*sizeof(np.complex128_t))
	if Q is not None:
		for ii in range(m):
			Q_inv[ii] = 1.0 / Q[ii]
		c_zvecmat(Q_inv, &U[0,0], m, nr)
		c_zvecmat(Q_inv, &V[0,0], m, nr)
	free(Q_inv)

	return U, S, V

@cr('RES.run_new')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def run_new(object X, object w, object r, int remove_mean=True, real[:] Q=None):

	if isinstance(X, list):
		if X[0].dtype == np.double:
			return _zrun_new(X,np.complex128(w),np.double(r),remove_mean, Q)
		else:
			return _crun_new(X,np.complex64(w),np.float32(r),remove_mean, Q)
	else:
		if X[0].dtype == np.double:
			return _zrun_old(X,np.complex128(w),np.double(r),remove_mean, Q)
		else:
			return _crun_old(X,np.complex64(w),np.float32(r),remove_mean, Q)