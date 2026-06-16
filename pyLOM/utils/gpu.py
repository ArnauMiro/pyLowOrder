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
		Setup the GPU to be used.

		Args:
			id: The MPI Rank.
			gpu_per_node: The number of GPUs available per node.
		'''
		local_id = int(id%gpu_per_node)
		cp.cuda.Device(local_id).use()

	def gpu_warmup():
		'''
		Two small activations of the matmul and QR
		algorithms allowing to setup the CUDA context
		'''
		# matmul
		A = cp.array([
			[1 , 1, 1],
			[-5, 2, 4],
			[1 , 5, 3],
			[4 , 3, 8],
			[6 , 4, 2]], 
		cp.float32, order = 'C')
		B = cp.array([
			[1 , 1, 1, 2],
			[-5, 2, 4, 5],
			[1 , 5, 3, 4]], 
		cp.float32, order = 'C')
		C = cp.matmul(A,B)
		# QR and SVD
		A = cp.array([[1,2],[3,4],[5,6],[7,8]],dtype=cp.float32,order='C')
		Q,R   = cp.linalg.qr(A)
		U,S,V = cp.linalg.svd(A)
		# matmul, QR and SVD - bigger size
		A = cp.random.rand(1000,100).astype(np.float32)
		B = cp.random.rand(100,1000).astype(np.float32)
		C = cp.matmul(A,B)
		Q,R   = cp.linalg.qr(A)
		U,S,V = cp.linalg.svd(A)

	def gpu_to_cpu(X):
		'''
		Move an array from GPU to CPU.

		Args:
			X (np.ndarray or cp.ndarray): The array to return.

		Returns:
			numpy.ndarray:
				Converted array on host memory.
				
		'''
		return cp.asnumpy(X) if type(X) is cp.ndarray else X

	def cpu_to_gpu(X):
		'''
		Move an array from GPU to CPU.

		Args:
			X (np.ndarray or cp.ndarray): The array to return.

		Returns:
			cupy.ndarray:
				Converted array on selected device.
		'''
		return cp.asarray(X) if type(X) is not cp.ndarray else X
	
	def ascontiguousarray(X):
		'''
		Returns a C-contiguous array.

		Args:
			X (np.ndarray or cp.ndarray): The array to return.

		Returns:
			np.ndarray or cp.ndarray:
				``X`` if no copy is required, otherwise a copy of ``X``.
		'''
		p = cp if type(X) is cp.ndarray else np
		return p.ascontiguousarray(X)
	
	def from_dlpack(X):
		'''
		Returns from_dlpack from the cupy package.

		Args:
			X : Array

		Returns:
			cp.from_dlpack(X)
		'''
		return cp.from_dlpack(X)

except:
	import numpy as cp

	def gpu_device(id=MPI_RANK,gpu_per_node=4):
		'''
		Setup the GPU to be used.

		Args:
			id: The MPI Rank.
			gpu_per_node: The number of GPUs available per node.
		'''
		raiseWarning('cupy not available! GPU version deactivated!')

	def gpu_warmup():
		'''
		Two small activations of the matmul and QR
		algorithms allowing to setup the CUDA context
		'''
		pass

	def gpu_to_cpu(X):
		'''
		Move an array from GPU to CPU.

		Args:
			X (np.ndarray or cp.ndarray): The array to return.

		Returns:
			numpy.ndarray:
				Converted array on host memory.
				
		'''
		return X

	def cpu_to_gpu(X):
		'''
		Move an array from GPU to CPU.

		Args:
			X (np.ndarray or cp.ndarray): The array to return.

		Returns:
			cupy.ndarray:
				Converted array on selected device.
		'''
		return X

	def ascontiguousarray(X):
		'''
		Returns a C-contiguous array.

		Args:
			X (np.ndarray or cp.ndarray): The array to return.

		Returns:
			np.ndarray or cp.ndarray:
				``X`` if no copy is required, otherwise a copy of ``X``.
		'''
		return np.ascontiguousarray(X)

	def from_dlpack(X):
		'''
		Returns from_dlpack from the cupy package.

		Args:
			X : Array

		Returns:
			cp.from_dlpack(X)
		'''
		return X
