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

from .cfuncs    cimport c_sRMSE, c_sRMSE_relative, c_sMAE, c_sr2, c_sMRE_array
from .cfuncs    cimport c_dRMSE, c_dRMSE_relative, c_dMAE, c_dr2, c_dMRE_array
from ..utils.cr  import cr

ctypedef fused real:
	float
	double

ctypedef fused realM:
	float[:]
	double[:]
	float[:,:]
	double[:,:]


@cr('math.RMSE')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def RMSE(realM A, realM B, int relative=True):
	r'''
	Compute the root mean square error between A and B

	Args:
		A (np.ndarray).
		B (np.ndarray).
		relative (bool, optional): default(``True``).

	Returns:
		(float): Root mean square error.
	'''
	cdef int m = A.shape[0], n = A.shape[1] if A.ndim > 1 else 1
	if realM is double[:]:
		return c_dRMSE_relative(&A[0],&B[0],m,n) if relative else c_dRMSE(&A[0],&B[0],m,n)
	elif realM is double[:,:]:
		return c_dRMSE_relative(&A[0,0],&B[0,0],m,n) if relative else c_dRMSE(&A[0,0],&B[0,0],m,n)
	elif realM is float[:]:
		return c_sRMSE_relative(&A[0],&B[0],m,n) if relative else c_sRMSE(&A[0],&B[0],m,n)
	else:
		return c_sRMSE_relative(&A[0,0],&B[0,0],m,n) if relative else c_sRMSE(&A[0,0],&B[0,0],m,n)

@cr('math.MAE')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def MAE(realM A, realM B):
	r'''
	Compute mean absolute error between A and B

	Args:
		A (np.ndarray).
		B (np.ndarray).

	Returns:
		(float): Mean absolute error.
	'''
	cdef int m = A.shape[0], n = A.shape[1] if A.ndim > 1 else 1
	if realM is double[:]:
		return c_dMAE(&A[0],&B[0],m,n)
	elif realM is double[:,:]:
		return c_dMAE(&A[0,0],&B[0,0],m,n) 
	elif realM is float[:]:
		return c_sMAE(&A[0],&B[0],m,n)
	else:
		return c_sMAE(&A[0,0],&B[0,0],m,n)

@cr('math.r2')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def r2(realM A, realM B):
	r'''
	Compute r2 score between A and B

	Args:
		A (np.ndarray).
		B (np.ndarray).

	Returns:
		(float): r2 score .
	'''
	cdef int m = A.shape[0], n = A.shape[1] if A.ndim > 1 else 1
	if realM is double[:]:
		return c_dr2(&A[0],&B[0],m,n)
	elif realM is double[:,:]:
		return c_dr2(&A[0,0],&B[0,0],m,n)
	elif realM is float[:]:
		return c_sr2(&A[0],&B[0],m,n)
	else:
		return c_sr2(&A[0,0],&B[0,0],m,n)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _sMRE_array(float[:,:] A, float[:,:] B, int axis=1):
	'''
	Mean relative error computed along a certain axis of the array.
	'''
	cdef int m = A.shape[0], n = A.shape[1], ldim = m if axis == 0 else n
	cdef np.ndarray[np.float32_t,ndim=1] out = np.zeros((ldim,),dtype=np.float32)
	c_sMRE_array(&out[0],&A[0,0],&B[0,0],m,n,axis)
	return out

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dMRE_array(double[:,:] A, double[:,:] B, int axis=1):
	'''
	Mean relative error computed along a certain axis of the array.
	'''
	cdef int m = A.shape[0], n = A.shape[1], ldim = m if axis == 0 else n
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((ldim,),dtype=np.double)
	c_dMRE_array(&out[0],&A[0,0],&B[0,0],m,n,axis)
	return out

@cr('math.MRE_array')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def MRE_array(real[:,:] A, real[:,:] B, int axis=1):
	r'''
	Mean relative error computed along a certain axis of the array.

	Args:
		A (np.ndarray): original field.
		B (np.ndarray): field which we want to compute the MRE of.
		axis (int, optional): along which axis the MRE will be computed (default ``1``).

	Returns:
		(np.ndarray): Mean relative error.
	'''
	if real is double:
		return _dMRE_array(A,B,axis)
	else:
		return _sMRE_array(A,B,axis)




