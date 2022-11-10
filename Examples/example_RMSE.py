#!/usr/bin/env python
#
# Example of SVD utility so that it returns
# the same output as MATLAB.
#
# Last revision: 09/07/2021
from __future__ import print_function, division

import numpy as np
import pyLOM


## Define matrices to compare
#A = np.array([[1,2],[3,4],[5,6],[7,8]],dtype=np.double,order='C')
#B = np.array([[1,6],[3,4],[10,6],[7,8]],dtype=np.double,order='C')

A = np.load('Xdmd.npy')
B = np.load('X.npy')

## Compute RMSE
r = pyLOM.math.RMSE(A, B)
print('RMSE is: ', r)
