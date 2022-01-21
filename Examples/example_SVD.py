#!/usr/bin/env python
#
# Example of SVD utility so that it returns
# the same output as MATLAB.
#
# Last revision: 09/07/2021
from __future__ import print_function, division

import numpy as np
import pyLOM


## Define matrix A 4x2
A = np.array([[1,2],[3,4],[5,6],[7,8]],dtype=np.double,order='C')


## Run SVD from numpy
pyLOM.cr_start('SVD numpy',0)
U, S, V = np.linalg.svd(A,full_matrices=False)
pyLOM.cr_stop('SVD numpy',0)
print('Numpy:')
print(U.shape,U)
print(S.shape,S)
print(V.shape,V)


## Run SVD from POD
U, S, V = pyLOM.math.svd(A)
print('pyLOM SVD:')
print(U.shape,U)
print(S.shape,S)
print(V.shape,V)

U, S, V = pyLOM.math.tsqr_svd(A)
print('pyLOM TSQR SVD:')
print(U.shape,U)
print(S.shape,S)
print(V.shape,V)


pyLOM.cr_info()