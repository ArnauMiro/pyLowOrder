#!/usr/bin/env python
#
# Example how to perform the transpose.
#
# Last revision: 01/02/2023
from __future__ import print_function, division

import numpy as np
import pyLOM

# Test matrix
A = np.array([
	[1, 6, 2, 9],
    [-5, 3, 4, -8],
    [4, 7, 8, 3]],dtype = np.double)
print('Matrix to transpose: \n', A)

# Cholesky decomposition from NumPy
Bnp = np.transpose(A)
print('Transpose NumPy: \n', Bnp)

# Cholesky decomposition from pyLOM:
BpyLOM = pyLOM.math.transpose(A)
print('Transpose pyLOM: \n', BpyLOM)

pyLOM.cr_info()