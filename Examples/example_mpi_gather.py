#!/usr/bin/env python
#
# Example how to perform parallel mpi_gather.
#
# Assume we have a matrix split with different M 
# and the same N scattered among p processors.
#
# We wish to allgather 
#
# Last revision: 18/03/2025
from __future__ import print_function, division

import numpy as np
import pyLOM

import pyLOM.utils
from pyLOM.utils.mpi import MPI_RANK, MPI_SIZE


## Define the global matrix size for this example 
# Sizes to be workable on a small machine
#M = 2000
M = 1440*721*(MPI_RANK+1)
N = 100


## Split the number of rows (M) between the number of processors
istart, iend = pyLOM.utils.worksplit(0,M,MPI_RANK,nWorkers=MPI_SIZE)
Mi = iend-istart # Local M for each processor


## Generate a large enough random matrix
Ai = np.random.rand(Mi,N).astype(np.float32) # Setting type to float or double
pyLOM.utils.pprint(-1,'Ai shape:',Ai.shape)


## All gather matrix A to all processors
A = pyLOM.utils.mpi_gather(Ai,0,all=True)
pyLOM.utils.pprint(-1,'A shape:',A.shape)


pyLOM.cr_info()