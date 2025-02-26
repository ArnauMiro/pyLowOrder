#!/bin/env python
# 
# Test pyLOM multi-gpu using cupy
#
# Last revision: 26/02/2025
from __future__ import print_function, division

import numpy as np, cupy as cp
import pyLOM

from pyLOM.utils.mpi import MPI_SIZE

pyLOM.gpu_device(gpu_per_node=4) # MN5 has 4 GPU per node


# Generate a large enough random matrix on the GPU
Ai = cp.random.rand(int(10e6),2000).astype(cp.float32)

# Run pyLOM TSQR-SVD algorithm
Ui, S, V = pyLOM.math.tsqr_svd(Ai)

# Print results
pyLOM.pprint(0,f'RUNNING WITH {MPI_SIZE} GPUs')
pyLOM.cr_info()
