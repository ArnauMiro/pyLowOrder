#!/usr/bin/env python
#
# pyLOM, utils.
#
# GPU setup routines
#
# Last rev: 14/02/2025
from __future__ import print_function, division

import cupy
from .mpi import MPI_RANK


def gpu_device(id=MPI_RANK,gpu_per_node=4):
	'''
	Setup the GPU to be used
	'''
	local_id = int(cupy.mod(id,gpu_per_node))
	print(local_id)
	cupy.cuda.Device(local_id).use()