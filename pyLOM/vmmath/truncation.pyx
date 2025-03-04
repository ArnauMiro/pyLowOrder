#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - truncation.
#
# Last rev: 27/10/2021

cimport cython
cimport numpy as np

import numpy as np

from libc.math  cimport fabs
from .cfuncs    cimport real, c_svector_norm, c_dvector_norm, c_senergy, c_denergy
from ..utils.cr  import cr


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef int _scompute_truncation_residual(float[:] S, float r):
	'''
	Compute the truncation residual.
	r must be a float precision (r<1) where:
		- r > 0: target residual
		- r < 0: fraction of cumulative energy to retain
	'''
	cdef int ii, N = 0, nS = S.shape[0]
	cdef float normS, accumulative
	if r > 0:
		normS = c_svector_norm(&S[0],0,nS)
		for ii in range(nS):
			accumulative = c_svector_norm(&S[0],ii,nS)/normS
			if accumulative < r: break
			N += 1
	else:
		r = fabs(r)
		normS = c_svector_norm(&S[0],0,nS)
		accumulative = 0
		for ii in range(nS):
			accumulative += S[ii]/normS
			N += 1		
			if accumulative > r: break
	return N

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef int _dcompute_truncation_residual(double[:] S, double r):
	'''
	Compute the truncation residual.
	r must be a float precision (r<1) where:
		- r > 0: target residual
		- r < 0: fraction of cumulative energy to retain
	'''
	cdef int ii, N = 0, nS = S.shape[0]
	cdef double normS, accumulative
	if r > 0:
		normS = c_dvector_norm(&S[0],0,nS)
		for ii in range(nS):
			accumulative = c_dvector_norm(&S[0],ii,nS)/normS
			if accumulative < r: break
			N += 1
	else:
		r = fabs(r)
		normS = c_dvector_norm(&S[0],0,nS)
		accumulative = 0
		for ii in range(nS):
			accumulative += S[ii]/normS
			N += 1		
			if accumulative > r: break
	return N

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def compute_truncation_residual(real[:] S, real r):
	'''
	Compute the truncation residual.
	r must be a float precision (r<1) where:
		- r > 0: target residual
		- r < 0: fraction of cumulative energy to retain
	'''
	if real is double:
		return _dcompute_truncation_residual(S,r)
	else:
		return _scompute_truncation_residual(S,r)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef float _senergy(float[:,:] A, float[:,:] B):
	'''
	Compute RMSE between X_POD and X
	'''
	cdef int m = A.shape[0], n = B.shape[1]
	cdef float Ek = 0.
	Ek = c_senergy(&A[0,0],&B[0,0],m,n)
	return Ek

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef double _denergy(double[:,:] A, double[:,:] B):
	'''
	Compute RMSE between X_POD and X
	'''
	cdef int m = A.shape[0], n = B.shape[1]
	cdef double Ek = 0.
	Ek = c_denergy(&A[0,0],&B[0,0],m,n)
	return Ek

@cr('math.energy')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def energy(real[:,:] A, real[:,:] B):
	'''
	Compute RMSE between X_POD and X
	'''
	if real is double:
		return _denergy(A,B)
	else:
		return _senergy(A,B)