#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - averaging.
#
# Last rev: 11/04/2025

cimport cython
cimport numpy as np

import numpy as np

from libc.stdlib cimport malloc, free
from .cfuncs     cimport real, c_stemporal_mean, c_dtemporal_mean, c_ssubtract_mean, c_dsubtract_mean, c_stemporal_variance, c_dtemporal_variance, c_snorm_variance, c_dnorm_variance
from ..utils.cr   import cr


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=1] _stemporal_mean(float[:,::1] X):
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
cdef np.ndarray[np.double_t,ndim=1] _dtemporal_mean(double[:,::1] X):
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
def temporal_mean(real[:,::1] X):
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
cdef np.ndarray[np.float32_t,ndim=1] _stemporal_variance(float[:,::1] X, float[::1] X_mean):
	'''
	Temporal variance of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.float32_t,ndim=1] out = np.zeros((m,),dtype=np.float32)
	# Compute temporal variance
	c_stemporal_variance(&out[0],&X[0,0],&X_mean[0],m,n)
	# Return
	return out

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] _dtemporal_variance(double[:,::1] X, double[::1] X_mean):
	'''
	Temporal variance of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((m,),dtype=np.double)
	# Compute temporal variance
	c_dtemporal_variance(&out[0],&X[0,0],&X_mean[0],m,n)
	# Return
	return out

@cr('math.temporal_mean')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def temporal_variance(real[:,::1] X, real[::1] X_mean):
	r'''
	Temporal variance of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.

	Args:
		X (numpy.ndarray): Snapshot matrix (m,n).

	Returns:
		numpy.ndarray: Averaged snapshot matrix (m,).
	'''
	if real is double:
		return _dtemporal_variance(X, X_mean)
	else:
		return _stemporal_variance(X, X_mean)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _ssubtract_mean(float[:,::1] X, float[::1] X_mean):
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
cdef np.ndarray[np.double_t,ndim=2] _dsubtract_mean(double[:,::1] X, double[::1] X_mean):
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
def subtract_mean(real[:,::1] X, real[::1] X_mean):
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

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _snorm_variance(float[:,::1] X, float[::1] X_mean, float[::1] X_var):
	r'''
	Computes out(m,n) = (X(m,n) - X_mean(m))/X_var(m) where m is the spatial coordinates
	and n is the number of snapshots.

	Args:
		X (numpy.ndarray): Snapshot matrix (m,n).
		X_mean (numpy.ndarray): Averaged snapshot matrix (m,)
		X_var (numpy.ndarray): Variance of snapshot matrix (m,)

	Returns:
		numpy.ndarray: Snapshot matrix without the average(m,n).
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] out2 = np.zeros((m,n),dtype=np.float32)
	out = <float*>malloc(m*n*sizeof(float))
	# Compute substract temporal mean
	c_ssubtract_mean(out,&X[0,0],&X_mean[0],m,n)
	# Normalize variance
	c_snorm_variance(&out2[0,0],out,&X_var[0],m,n)
	free(out)
	# Return
	return out2

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dnorm_variance(double[:,::1] X, double[::1] X_mean, double[::1] X_var):
	r'''
	Computes out(m,n) = (X(m,n) - X_mean(m))/X_var(m) where m is the spatial coordinates
	and n is the number of snapshots.

	Args:
		X (numpy.ndarray): Snapshot matrix (m,n).
		X_mean (numpy.ndarray): Averaged snapshot matrix (m,)
		X_var (numpy.ndarray): Variance of snapshot matrix (m,)

	Returns:
		numpy.ndarray: Snapshot matrix without the average(m,n).
	'''
	cdef int m = X.shape[0], n = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] out2 = np.zeros((m,n),dtype=np.double)
	out = <double*>malloc(m*n*sizeof(double))
	# Compute substract temporal mean
	c_dsubtract_mean(out,&X[0,0],&X_mean[0],m,n)
	# Normalize variance
	c_dnorm_variance(&out2[0,0],out,&X_var[0],m,n)
	free(out)
	# Return
	return out2

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def norm_variance(real[:,::1] X, real[::1] X_mean, real[::1] X_var):
	r'''
	Computes out(m,n) = (X(m,n) - X_mean(m))/X_var(m) where m is the spatial coordinates
	and n is the number of snapshots.

	Args:
		X (numpy.ndarray): Snapshot matrix (m,n).
		X_mean (numpy.ndarray): Averaged snapshot matrix (m,)
		X_var (numpy.ndarray): Variance of snapshot matrix (m,)

	Returns:
		numpy.ndarray: Snapshot matrix without the average(m,n).
	'''
	if real is double:
		return _dnorm_variance(X,X_mean,X_var)
	else:
		return _snorm_variance(X,X_mean,X_var)