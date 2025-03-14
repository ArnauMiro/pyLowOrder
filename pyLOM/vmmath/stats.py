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
def RMSE(A,B,relative=True):
	'''
	Compute RMSE between A and B
	'''
	p = cp if type(A) is cp.ndarray else np
	diff  = (A-B)
	sum1g = mpi_reduce(p.sum(diff*diff),op='sum',all=True)
	sum2g = mpi_reduce(p.sum(A*A),op='sum',all=True) if relative else p.prod(mpi_reduce(p.array(A.shape),op='sum',all=True))
	rmse  = np.sqrt(sum1g/sum2g)
	return rmse

@cr('math.MAE')
def MAE(A,B):
	'''
	Compute MAE between A and B
	'''
	p = cp if type(A) is cp.ndarray else np
	diff  = p.abs(A-B)
	sum1g = mpi_reduce(p.sum(diff),op='sum',all=True)
	sum2g = p.prod(mpi_reduce(p.array(A.shape),op='sum',all=True))
	mae   = sum1g/sum2g
	return mae

@cr('math.r2')
def r2(A,B):
	'''
	Compute r2 score between A and B
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