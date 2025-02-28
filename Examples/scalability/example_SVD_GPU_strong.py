#!/bin/env python
# 
# Test pyLOM multi-gpu using cupy
# Efficiency test, all GPUs have a different size
#
# Last revision: 26/02/2025
from __future__ import print_function, division

import numpy as np, cupy as cp
import pyLOM

from pyLOM.utils.mpi import MPI_RANK, MPI_SIZE

pyLOM.gpu_device(gpu_per_node=4) # MN5 has 4 GPU per node

# Define the matrix size
N = int(10e5)
istart, iend = pyLOM.utils.worksplit(0,N,MPI_RANK,nWorkers=MPI_SIZE)
Ni = iend-istart

# Generate random matrix
Ai = cp.random.rand(Ni,2000).astype(cp.float32)

# Run pyLOM TSQR-SVD algorithm
Ui, S, V = pyLOM.math.tsqr_svd(Ai)

# Print results
pyLOM.pprint(0,f'RUNNING WITH {MPI_SIZE} GPUs')
pyLOM.cr_info()
