#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - regression.
#
# Last rev: 27/10/2021

cimport cython
cimport numpy as np

import numpy as np

from .cfuncs   cimport real, c_sleast_squares, c_sridge_regression, c_dleast_squares, c_dridge_regression

from ..utils.cr import cr


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=1] _sleast_squares(float[:,:] A, float[:] b):
	'''
	Least squares regression
	(A^T * A)^-1 * A^T * b
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.float32_t,ndim=1] out = np.zeros((n,),dtype=np.float32)
	c_sleast_squares(&out[0],&A[0,0],&b[0],m,n)
	return out

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] _dleast_squares(double[:,:] A, double[:] b):
	'''
	Least squares regression
	(A^T * A)^-1 * A^T * b
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((n,),dtype=np.double)
	c_dleast_squares(&out[0],&A[0,0],&b[0],m,n)
	return out

@cr('math.least_squares')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def least_squares(real[:,:] A, real[:] b):
	'''
	Least squares regression
	(A^T * A)^-1 * A^T * b
	'''
	if real is double:
		return _dleast_squares(A,b)
	else:
		return _sleast_squares(A,b)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=1] _sridge_regression(float[:,:] A, float[:] b, float lam):
	'''
	Least squares regression
	(A^T * A)^-1 * A^T * b
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.float32_t,ndim=1] out = np.zeros((n,),dtype=np.float32)
	c_sridge_regression(&out[0],&A[0,0],&b[0],lam,m,n)
	return out

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] _dridge_regression(double[:,:] A, double[:] b, double lam):
	'''
	Least squares regression
	(A^T * A)^-1 * A^T * b
	'''
	cdef int m = A.shape[0], n = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((n,),dtype=np.double)
	c_dridge_regression(&out[0],&A[0,0],&b[0],lam,m,n)
	return out

@cr('math.ridge_regresion')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def ridge_regresion(real[:,:] A, real[:] b, real lam):
	'''
	Ridge regression
	'''
	if real is double:
		return _dridge_regression(A,b,lam)
	else:
		return _sridge_regression(A,b,lam)