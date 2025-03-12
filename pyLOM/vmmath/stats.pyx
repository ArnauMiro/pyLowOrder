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

from .cfuncs    cimport c_sRMSE, c_sRMSE_relative, c_dRMSE, c_dRMSE_relative, c_sMAE, c_dMAE, c_sr2, c_dr2
from ..utils.cr  import cr

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
	'''
	Compute RMSE between A and B
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
	'''
	Compute MAE between A and B
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
	'''
	Compute MAE between A and B
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