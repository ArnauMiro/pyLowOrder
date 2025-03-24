#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - averaging.
#
# Last rev: 27/10/2021

cimport cython
cimport numpy as np

import numpy as np

from .cfuncs    cimport real, c_stemporal_mean, c_dtemporal_mean, c_ssubtract_mean, c_dsubtract_mean
from ..utils.cr  import cr


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=1] _stemporal_mean(float[:,:] X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.float32_t,ndim=1] out = np.zeros((m,),dtype=np.float32)
	# Compute temporal mean
	c_stemporal_mean(&out[0],&X[0,0],m,n)
	# Return
	return out

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] _dtemporal_mean(double[:,:] X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((m,),dtype=np.double)
	# Compute temporal mean
	c_dtemporal_mean(&out[0],&X[0,0],m,n)
	# Return
	return out

@cr('math.temporal_mean')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def temporal_mean(real[:,:] X):
	r'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.

	Args:
		X (numpy.ndarray): Snapshot matrix (m,n).

	Returns:
		numpy.ndarray: Averaged snapshot matrix (m,).
	'''
	if real is double:
		return _dtemporal_mean(X)
	else:
		return _stemporal_mean(X)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _ssubtract_mean(float[:,:] X, float[:] X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] out = np.zeros((m,n),dtype=np.float32)
	# Compute substract temporal mean
	c_ssubtract_mean(&out[0,0],&X[0,0],&X_mean[0],m,n)
	# Return
	return out

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dsubtract_mean(double[:,:] X, double[:] X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] out = np.zeros((m,n),dtype=np.double)
	# Compute substract temporal mean
	c_dsubtract_mean(&out[0,0],&X[0,0],&X_mean[0],m,n)
	# Return
	return out

@cr('math.subtract_mean')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def subtract_mean(real[:,:] X, real[:] X_mean):
	r'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.

	Args:
		X (numpy.ndarray): Snapshot matrix (m,n).
		X_mean (numpy.ndarray): Averaged snapshot matrix (m,)

	Returns:
		numpy.ndarray: Snapshot matrix without the average(m,n).
	'''
	if real is double:
		return _dsubtract_mean(X,X_mean)
	else:
		return _ssubtract_mean(X,X_mean)