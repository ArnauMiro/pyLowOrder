#!/usr/bin/env python
#
# Example of SVD utility so that it returns
# the same output as MATLAB.
#
# Last revision: 09/07/2021
from __future__ import print_function, division

import numpy as np
import pyLOM
from pyLOM.utils.parall import mpi_scatter

## Define matrices to compare
A  = np.array([[1,2],[3,4],[5,6],[7,8]],dtype=np.double,order='C')
Ai = mpi_scatter(A,root=0,do_split=True)
B  = np.array([[1,6],[3,4],[10,6],[7,8]],dtype=np.double,order='C')
Bi = mpi_scatter(B,root=0,do_split=True)

## Compute RMSE
r = pyLOM.math.RMSE(A, B)
pyLOM.pprint(0, 'RMSE is: ', r, flush = True)
