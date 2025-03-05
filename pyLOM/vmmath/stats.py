#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - statistics.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.cr     import cr
from ..utils.parall import mpi_reduce


@cr('math.RMSE')
def RMSE(A,B):
	'''
	Compute RMSE between A and B
	'''
	diff  = (A-B)
	sum1g = mpi_reduce(np.sum(diff*diff),op='sum',all=True)
	sum2g = np.prod(mpi_reduce(np.array(A.shape),op='sum',all=True))
#	sum2g = mpi_reduce(np.sum(A*A),op='sum',all=True)
	rmse  = np.sqrt(sum1g/sum2g)
	return rmse

@cr('math.MAE')
def MAE(A,B):
	'''
	Compute MAE between A and B
	'''
	diff  = np.abs(A-B)
	sum1g = mpi_reduce(np.sum(diff),op='sum',all=True)
	sum2g = np.prod(mpi_reduce(np.array(A.shape),op='sum',all=True))
	mae   = sum1g/sum2g
	return mae

@cr('math.r2')
def r2(A,B):
	'''
	Compute r2 score between A and B
	'''
	num  = (A-B)
	numg = mpi_reduce(np.sum(num*num),op='sum',all=True)
	sumg = mpi_reduce(np.sum(A),op='sum',all=True)
	sum2g = np.prod(mpi_reduce(np.array(A.shape),op='sum',all=True))
	den  = A - sumg/sum2g
	deng = mpi_reduce(np.sum(den*den),op='sum',all=True)
	r2   = 1. - numg/deng
	return r2