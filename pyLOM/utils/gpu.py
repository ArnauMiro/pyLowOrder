#!/usr/bin/env python
#
# pyLOM, utils.
#
# GPU setup routines
#
# Last rev: 14/02/2025
from __future__ import print_function, division

import numpy as np

from .mpi    import MPI_RANK
from .errors import raiseWarning

try:
	import cupy as cp

	def gpu_device(id=MPI_RANK,gpu_per_node=4):
		'''
		Setup the GPU to be used
		'''
		local_id = int(id%gpu_per_node)
		cp.cuda.Device(local_id).use()

	def gpu_to_cpu(X):
		'''
		Move an array from GPU to CPU
		'''
		return cp.asnumpy(X) if type(X) is cp.ndarray else X

	def cpu_to_gpu(X):
		'''
		Move an array from GPU to CPU
		'''
		return cp.asarray(X) if type(X) is not cp.ndarray else X
	
	def ascontiguousarray(X):
		p = cp if type(X) is cp.ndarray else np
		return p.ascontiguousarray(X)

except:
	import numpy as cp

	def gpu_device(id=MPI_RANK,gpu_per_node=4):
		'''
		Setup the GPU to be used
		'''
		raiseWarning('cupy not available! GPU version deactivated!')

	def gpu_to_cpu(X):
		'''
		Move an array from GPU to CPU
		'''
		return X

	def cpu_to_gpu(X):
		'''
		Move an array from GPU to CPU
		'''
		return X

	def ascontiguousarray(X):
		return np.ascontiguousarray(X)