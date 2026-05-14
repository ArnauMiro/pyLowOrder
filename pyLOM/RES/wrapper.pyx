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
from ..vmmath.cfuncs cimport c_zsvd, c_zdagger, c_zmatmul, c_zmatmulp, c_cvecmat, c_zcholesky, c_cinverse

from ..utils.cr       import cr, cr_start, cr_stop
from ..utils.errors   import raiseError


## RES run method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _srun(np.complex64_t[:,:] Phi, float[:] delta, float[:] freq, float f, float[:,:] Q=None):

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
	cdef np.complex64_t *Fhat
	Phi_dagger = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
	Fhat = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	c_cdagger(Phi, Phi_dagger, m, n)
	if Q is None:
		c_cmatmulp(Fhat, Phi_dagger, Phi, n, n, m)
	else:
		cdef np.complex64_t *Phi_aux
		Phi_aux = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
		memcpy(Phi_aux, &Phi[0,0], m*n*sizeof(np.complex64_t))
		c_cvecmat(Q, Phi_aux, m, n) # Phi_aux is overwritten
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
	memcpy(Fhat_inv, Fhat, m*n*sizeof(np.complex64_t))
	retval = c_cinv(Fhat_inv, n, 'L')
	if not retval == 0: raiseError('Problems computing the Inverse!')
	cr_stop('RES.Choleski', 0)

	# Compute Hhat
	cr_start('RES.Hhat', 0)
	cdef np.complex64_t *Hhat
	cdef np.complex64_t *Fhat_aux
	Hhat = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	Fhat_aux = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	memcpy(Fhat_aux, Fhat_inv, m*n*sizeof(np.complex64_t))
	c_cvecmat(H, Fhat_aux, n, n)
	c_cmatmul(Hhat, Fhat, Fhat_aux, n, n, n)
	free(H)
	ree(H_aux)
	free(Fhat)
	free(Fhat_aux)
	cr_stop('RES.Hhat', 0)

	# Compute the svd
	cr_start('RES.svd', 0)
	cdef np.complex64_t *U
	cdef float *S
	cdef np.complex64_t *V
	cdef np.complex64_t *Vt
	U = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	S = <float*>malloc(n*sizeof(float))
	V = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	Vt = <np.complex64_t*>malloc(n*n*sizeof(np.complex64_t))
	c_csvd(U, S, Vt, Hhat, n, n)
	c_cdagger(Vt, V, n, n)
	free(Hhat)
	free(Vt)
	cr_stop('RES.svd', 0)

	# Compute the projection
	cr_start('RES.projection', 0)
	cdef np.complex64_t *U_res
	cdef np.complex64_t *V_res
	cdef np.complex64_t *U_aux
	cdef np.complex64_t *V_aux
	U_res = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
	V_res = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
	U_aux = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
	V_aux = <np.complex64_t*>malloc(m*n*sizeof(np.complex64_t))
	c_cmatmul(U_aux, Fhat_inv, U, n, n, n)
	c_cmatmul(V_aux, Fhat_inv, V, n, n, n)
	c_cmatmul(U_res, Phi, U_aux, m, n, n)
	c_cmatmul(V_res, Phi, V_aux, m, n, n)
	free(Fhat_inv)
	free(U)
	free(U_aux)
	free(V)
	free(V_aux)
	cr_stop('RES.projection', 0)

	return U_res, S, V_res