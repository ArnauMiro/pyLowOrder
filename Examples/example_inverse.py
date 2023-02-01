#!/usr/bin/env python
#
# Example how to perform Cholesky inverse.
#
# Last revision: 01/02/2023
from __future__ import print_function, division

import numpy as np
import pyLOM

# Test matrix
A = np.array([
	[1 + 1*1j, 1 + 6*1j, 1 + 2*1j],
    [-5 + 3*1j, 2 + 3*1j, 4 + 1*1j],
    [1 + 4*1j, 5 + 7*1j, 3 + 8*1j]])
B = np.matmul(A, np.transpose(np.conjugate(A)))
print('Matrix to factorize: \n', B)

# Cholesky decomposition from NumPy
Bchol = np.linalg.cholesky(B)
Binv  = np.linalg.inv(Bchol)

# Cholesky decomposition from pyLOM:
B = np.matmul(A, np.transpose(np.conjugate(A)))
BpyLOM  = pyLOM.math.cholesky(B)
BpyLOM2 = BpyLOM.copy()
BipyLOM = pyLOM.math.inv(BpyLOM2)

print('Cholesky NumPy: \n', Bchol)
print('Cholesky pyLOM: \n', BpyLOM)

print('Inverse NumPy: \n', Binv)
print('Inverse pyLOM: \n', BipyLOM)

print('Difference in Cholesky: \n', np.abs(BpyLOM - Bchol))
print('Difference in inverse: \n', np.abs(BipyLOM - Binv))

pyLOM.cr_info()
