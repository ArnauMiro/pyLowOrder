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


def gpu_device(id=MPI_RANK):
	'''
	Setup the GPU to be used
	'''
	cupy.cuda.Device(id).use()