#!/bin/env python
# 
# Test pyLOM multi-gpu using cupy
# Efficiency test, all GPUs have the same size matrix
#
#  Matrix size guidelines depending on the machine tested:
#
#### A node of MareNostrum 5 (4 NVIDIA H100 GPUs per node with 64 Gb of VRAM each) can handle the following sizes:
######## M = 8e6, N = 500
######## M = 4e6, N = 1000
######## M = 2e6, N = 2000
#
#### A single GPU of Juno 3 (48 Gb of VRAM) can handle the following sizes:
######## M = 8e6, N = 250
######## M = 4e6, N = 500
######## M = 2e6, N = 1000
#
#### Add any other tested machine ####
# Last revision: 26/02/2025
from __future__ import print_function, division

import numpy as np, cupy as cp
import pyLOM
import sys

from pyLOM.utils.mpi import MPI_SIZE

pyLOM.gpu_device(gpu_per_node=4) # MN5 has 4 GPU per node

# Define the global matrix size according to the capabilities of your machine
M = int(np.double(sys.argv[1]))
N = int(sys.argv[2])

# Generate a large enough random matrix on the GPU
Ai = cp.random.rand(M,N).astype(cp.float32)

# Run pyLOM TSQR-SVD algorithm
Ui, S, V = pyLOM.math.tsqr_svd(Ai)

# Print results
pyLOM.pprint(0,f'RUNNING WITH {MPI_SIZE} GPUs')
pyLOM.cr_info()
