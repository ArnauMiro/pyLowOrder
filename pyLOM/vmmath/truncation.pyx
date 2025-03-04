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

from .cfuncs    cimport real, c_senergy, c_denergy
from ..utils.cr  import cr


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