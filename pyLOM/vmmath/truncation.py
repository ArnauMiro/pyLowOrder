#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - truncation.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp
from .maths      import vector_norm, vector_sum
from ..utils     import cr_nvtx as cr, mpi_reduce


def compute_truncation_residual(S, r):
	'''
	Compute the truncation residual.
	r must be a float precision (r<1) where:
		- r > 0: target residual
		- r < 0: fraction of cumulative energy to retain
	'''
	N = 0
	if r > 0:
		normS = vector_norm(S,0)
		for ii in range(S.shape[0]):
			accumulative = vector_norm(S,ii)/normS
			if accumulative < r: break
			N += 1
	else:
		r = abs(r)
		normS = vector_sum(S,0)
		accumulative = 0
		for ii in range(S.shape[0]):
			accumulative += S[ii]/normS
			N += 1		
			if accumulative > r: break
	return N

@cr('math.energy')
def energy(original, rec):
	'''
	Compute reconstruction energy as in:
	Eivazi, H., Le Clainche, S., Hoyas, S., & Vinuesa, R. (2022). 
	Towards extraction of orthogonal and parsimonious non-linear modes from turbulent flows. 
	Expert Systems with Applications, 202, 117038.
	https://doi.org/10.1016
	'''
	p = cp if type(original) is cp.ndarray else np
	# Compute local sums
	local_num = p.sum((original - rec) ** 2)
	local_den = p.sum(original ** 2)

	# Use Allreduce to compute global sums and make them available on all ranks
	global_num = mpi_reduce(local_num,op='sum',all=True)
	global_den = mpi_reduce(local_den,op='sum',all=True)

	# Compute Ek (this will be identical on all ranks)
	return 1 - global_num / global_den