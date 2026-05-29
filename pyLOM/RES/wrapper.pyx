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
from libc.stdlib     cimport malloc, free
from libc.string     cimport memcpy, memset
from libc.math       cimport sqrt, log, atan2
from ..vmmath.cfuncs cimport real, real_complex
# from ..vmmath.cfuncs cimport 
# from ..vmmath.cfuncs cimport 
from ..vmmath.cfuncs cimport c_csvd, c_cdagger, c_cmatmul, c_cmatmulp, c_cvecmat, c_ccholesky, c_cinverse
from ..vmmath.cfuncs cimport c_zsvd, c_zdagger, c_zmatmul, c_zmatmulp, c_zvecmat, c_zcholesky, c_zinverse

from ..utils.cr       import cr, cr_start, cr_stop
from ..utils.errors   import raiseError


## RES run method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _crun(np.complex64_t[:,:] Phi, float[:] delta, float[:] freq, float f, float[:] Q=None):
	'''
    Resolvent Analysis of snapshot matrix X
    Inputs:
        - X[ndims*nmesh,n_temp_snapshots]: data matrix
        - delta: damping ratio of each mode
        - freq: frequency of each mode
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
		Omega[ii] = delta[ii] + J * freq[ii]
		H[ii] = 1 / (-J * f - Omega[ii])
	free(Omega)
	cr_stop('RES.resolvent_operator', 0)

	# Compute the Qhat (named Fhat for convenience)
	cr_start('RES.Qhat', 0)
	cdef np.complex64_t *Phi_dagger
	cdef np.complex64_t *Phi_aux
	cdef np.complex64_t *Fhat
	Phi_dagger = <np.complex64_t*>malloc(n*m*sizeof(np.complex64_t))
	Fhat = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	Phi_aux = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	cdef np.ndarray[np.complex64_t,ndim=1] Q_aux = np.zeros((m),dtype=np.complex64)
	c_cdagger(&Phi[0,0], Phi_dagger, m, n)
	if Q is None:
		c_cmatmulp(Fhat, Phi_dagger, &Phi[0,0], n, n, m)
	else:
		memcpy(Phi_aux, &Phi[0,0], m*n*sizeof(np.complex64_t))
		Q_aux = np.array(Q, dtype=np.complex64)
		c_cvecmat(&Q_aux[0], Phi_aux, m, n) # Phi_aux is overwritten
		c_cmatmulp(Fhat, Phi_dagger, Phi_aux, n, n, m)
	free(Phi_aux)
	free(Phi_dagger)
	cr_stop('RES.Qhat', 0)

	# Compute the Choleski decomposition
	cr_start('RES.Choleski', 0)
	retval = c_ccholesky(Fhat, n)
	if not retval == 0: raiseError('Problems computing Cholesky factorization!')
	cdef np.complex64_t *Fhat_inv
	Fhat_inv = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	memcpy(Fhat_inv, Fhat, n*n*sizeof(np.complex64_t))
	retval = c_cinverse(Fhat_inv, n, 'L')
	if not retval == 0: raiseError('Problems computing the Inverse!')
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
	# cdef np.complex64_t *U_res
	# cdef np.complex64_t *V_res
	cdef np.complex64_t *U_aux
	cdef np.complex64_t *V_aux
	# U_res = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
	# V_res = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
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
def _zrun(np.complex128_t[:,:] Phi, double[:] delta, double[:] freq, double f, double[:] Q=None):
	'''
    Resolvent Analysis of snapshot matrix X
    Inputs:
        - X[ndims*nmesh,n_temp_snapshots]: data matrix
        - delta: damping ratio of each mode
        - freq: frequency of each mode
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
		Omega[ii] = delta[ii] + J * freq[ii]
		H[ii] = 1 / (-J * f - Omega[ii])
	free(Omega)
	cr_stop('RES.resolvent_operator', 0)

	# Compute the Qhat (named Fhat for convenience)
	cr_start('RES.Qhat', 0)
	cdef np.complex128_t *Phi_dagger
	cdef np.complex128_t *Phi_aux
	cdef np.complex128_t *Fhat
	Phi_dagger = <np.complex128_t*>malloc(n*m*sizeof(np.complex128_t))
	Fhat = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	Phi_aux = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	cdef np.ndarray[np.complex128_t,ndim=1] Q_aux = np.zeros((m),dtype=np.complex128)
	c_zdagger(&Phi[0,0], Phi_dagger, m, n)
	if Q is None:
		c_zmatmulp(Fhat, Phi_dagger, &Phi[0,0], n, n, m)
	else:
		memcpy(Phi_aux, &Phi[0,0], m*n*sizeof(np.complex128_t))
		Q_aux = np.array(Q, dtype=np.complex128)
		c_zvecmat(&Q_aux[0], Phi_aux, m, n) # Phi_aux is overwritten
		c_zmatmulp(Fhat, Phi_dagger, Phi_aux, n, n, m)
	free(Phi_aux)
	free(Phi_dagger)
	cr_stop('RES.Qhat', 0)

	# Compute the Choleski decomposition
	cr_start('RES.Choleski', 0)
	retval = c_zcholesky(Fhat, n)
	if not retval == 0: raiseError('Problems computing Cholesky factorization!')
	cdef np.complex128_t *Fhat_inv
	Fhat_inv = <np.complex128_t*>malloc(n*n*sizeof(np.complex128_t))
	memcpy(Fhat_inv, Fhat, n*n*sizeof(np.complex128_t))
	retval = c_zinverse(Fhat_inv, n, 'L')
	if not retval == 0: raiseError('Problems computing the Inverse!')
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
	# cdef np.complex128_t *U_res
	# cdef np.complex128_t *V_res
	cdef np.complex128_t *U_aux
	cdef np.complex128_t *V_aux
	# U_res = <np.complex128_t*>malloc(m*n*sizeof(np.complex128_t))
	# V_res = <np.complex128_t*>malloc(m*n*sizeof(np.complex128_t))
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

def run(real_complex[:,:] Phi, real[:] delta, real[:] freq, real f, real[:] Q=None):
	'''
    Resolvent Analysis of snapshot matrix X
    Inputs:
        - X[ndims*nmesh,n_temp_snapshots]: data matrix
        - delta: damping ratio of each mode
        - freq: frequency of each mode
        - f: target frequency
        - Q: weighting matrix
    Returns:
        - U_res: response modes
        - S: emergy gains
        - V_res: forcing modes
    '''
	if real_complex is np.complex128_t:
		return _zrun(Phi, delta, freq, f, Q)
	else:
		return _crun(Phi, delta, freq, f, Q)
	# return _crun(Phi, delta, freq, f, Q)