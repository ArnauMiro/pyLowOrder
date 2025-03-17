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
from ..utils     import raiseError


@cr('math.RMSE')
def RMSE(A:np.ndarray,B:np.ndarray,relative:bool=True):
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
def MAE(A:np.ndarray,B:np.ndarray):
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
def r2(A:np.ndarray,B:np.ndarray):
	'''
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

def axiswise_r2(original:np.ndarray, rec:np.ndarray, axis:int=1):
	r'''
	Mean relative error computed along a certain axis of the array.

	Args:
		original (np.ndarray): original field.
		rec (np.ndarray): field which we want to compute the MRE of.
		axis (int, optional): along which axis the MRE will be computed (default ``1``).

	Returns:
		(np.ndarray): Mean relative error.
	'''
	# Compute local sums
	local_num = np.sum((original - rec) ** 2, axis=axis)
	local_den = np.sum(original ** 2, axis=axis)

	# Compute Mean Relative Error (this will be identical on all ranks)
	return local_num / local_den

def data_splitting(Nt:int, mode:str='reconstruct', seed:int=-1):
	r'''
	Generate random training, validation and test masks for a dataset of Nt samples.

	Args:
		Nt (int): number of data samples.
		mode (str, optional): type of splitting to perform (default: ``reconstruct``). In reconstruct mode all three datasets have samples along all the data range.
		seed (int, optional): (default: ``-1``).

	Returns:
		[(np.array), (np.array), (np.array)]: List of arrays containing the identifiers of the training, validation and test samples.
	'''
	np.random.seed(0) if seed < 0 else np.random.seed(seed)
	if mode =='reconstruct':
		tridx       = np.sort(np.random.choice(Nt, size=int(0.7*(Nt)), replace=False))
		mask        = np.ones(Nt)
		mask[tridx] = 0
		mask[0]     = 0
		mask[-1]    = 0
		vate_idx    = np.arange(0, Nt)[np.where(mask!=0)[0]]
		vaidx       = vate_idx[::2]
		teidx       = vate_idx[1::2]
	else:
		raiseError('Data split mode not implemented yet')
	return tridx, vaidx, teidx