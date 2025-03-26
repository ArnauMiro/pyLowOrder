#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - statistics.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp
from ..utils     import cr_nvtx as cr, mpi_reduce


@cr('math.RMSE')
def RMSE(A:np.ndarray,B:np.ndarray,relative:bool=True) -> float:
	r'''
	Compute the root mean square error between A and B

	Args:
		A (np.ndarray).
		B (np.ndarray).
		relative (bool, optional): default(``True``).

	Returns:
		(float): Root mean square error.
	'''
	p = cp if type(A) is cp.ndarray else np
	diff  = (A-B)
	sum1g = mpi_reduce(p.sum(diff*diff),op='sum',all=True)
	sum2g = mpi_reduce(p.sum(A*A),op='sum',all=True) if relative else p.prod(mpi_reduce(p.array(A.shape),op='sum',all=True))
	rmse  = np.sqrt(sum1g/sum2g)
	return rmse

@cr('math.MAE')
def MAE(A:np.ndarray,B:np.ndarray) -> float:
	r'''
	Compute mean absolute error between A and B

	Args:
		A (np.ndarray).
		B (np.ndarray).

	Returns:
		(float): Mean absolute error.
	'''
	p = cp if type(A) is cp.ndarray else np
	diff  = p.abs(A-B)
	sum1g = mpi_reduce(p.sum(diff),op='sum',all=True)
	sum2g = p.prod(mpi_reduce(p.array(A.shape),op='sum',all=True))
	mae   = sum1g/sum2g
	return mae

@cr('math.r2')
def r2(A:np.ndarray,B:np.ndarray) -> float:
	r'''
	Compute r2 score between A and B

	Args:
		A (np.ndarray).
		B (np.ndarray).

	Returns:
		(float): r2 score .
	'''
	p = cp if type(A) is cp.ndarray else np
	num  = (A-B)
	numg = mpi_reduce(p.sum(num*num),op='sum',all=True)
	sumg = mpi_reduce(p.sum(A),op='sum',all=True)
	sum2g = p.prod(mpi_reduce(p.array(A.shape),op='sum',all=True))
	den  = A - sumg/sum2g
	deng = mpi_reduce(p.sum(den*den),op='sum',all=True)
	r2   = 1. - numg/deng
	return r2

@cr('math.MRE_array')
def MRE_array(A:np.ndarray, B:np.ndarray, axis:int=1) -> np.ndarray:
	r'''
	Mean relative error computed along a certain axis of the array.

	Args:
		A (np.ndarray): original field.
		B (np.ndarray): field which we want to compute the MRE of.
		axis (int, optional): along which axis the MRE will be computed (default ``1``).

	Returns:
		(np.ndarray): Mean relative error.
	'''
	p = cp if type(A) is cp.ndarray else np
	# Compute local sums
	num  = (A-B)
	numg = p.sum(num*num,axis=axis)
	deng = p.sum(B*B,axis=axis)
	# Compute Mean Relative Error (this will be identical on all ranks)
	return numg/deng