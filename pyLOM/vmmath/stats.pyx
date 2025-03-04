#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - statistics.
#
# Last rev: 27/10/2021

cimport cython
cimport numpy as np

import numpy as np

from .cfuncs    cimport c_sRMSE, c_dRMSE
from ..utils.cr  import cr

ctypedef fused realM:
	float[:]
	double[:]
	float[:,:]
	double[:,:]


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef float _sRMSE(float *A, float *B, int m, int n):
	'''
	Compute RMSE between A and B
	'''
	return c_sRMSE(A,B,m,n)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef double _dRMSE(double *A, double *B, int m, int n):
	'''
	Compute RMSE between A and B
	'''
	return c_dRMSE(A,B,m,n)

@cr('math.RMSE')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def RMSE(realM A, realM B):
	'''
	Compute RMSE between A and B
	'''
	cdef int m = A.shape[0], n = A.shape[1] if A.ndim > 1 else 1
	if realM is double[:]:
		return _dRMSE(&A[0],&B[0],m,n)
	elif realM is double[:,:]:
		return _dRMSE(&A[0,0],&B[0,0],m,n) 
	elif realM is float[:]:
		return _sRMSE(&A[0],&B[0],m,n)
	else:
		return _sRMSE(&A[0,0],&B[0,0],m,n)