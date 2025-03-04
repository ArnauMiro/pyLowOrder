#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - truncation.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.cr     import cr
from ..utils.parall import mpi_reduce


@cr('math.energy')
def energy(original, rec):
	'''
	Compute reconstruction energy as in:
	Eivazi, H., Le Clainche, S., Hoyas, S., & Vinuesa, R. (2022). 
	Towards extraction of orthogonal and parsimonious non-linear modes from turbulent flows. 
	Expert Systems with Applications, 202, 117038.
	https://doi.org/10.1016
	'''
	# Compute local sums
	local_num = np.sum((original - rec) ** 2)
	local_den = np.sum(original ** 2)

	# Use Allreduce to compute global sums and make them available on all ranks
	global_num = mpi_reduce(local_num,op='sum',all=True)
	global_den = mpi_reduce(local_den,op='sum',all=True)

	# Compute Ek (this will be identical on all ranks)
	return 1 - global_num / global_den