#!/usr/bin/env python
#
# Example how to perform matmul with complex numbers.
#
# Last revision: 01/02/2023
from __future__ import print_function, division

import numpy as np
import pyLOM


## Testing normal matmul
A = np.array([
    [1 , 1, 1],
    [-5, 2, 4],
    [1 , 5, 3],
    [4 , 3, 8],
    [6 , 4, 2]], np.double, order = 'C')
B = np.array([
    [1 , 1, 1, 2],
    [-5, 2, 4, 5],
    [1 , 5, 3, 4]], np.double, order = 'C')

# NumPy matmul
Cnp = np.matmul(A, B)
print('NumPy: \n', Cnp)

# Complex matmul from pyLOM:
C = pyLOM.math.matmul(A, B)
print('pyLOM: \n', C)

print('Error: \n', C - Cnp)


## Test complex matmul
A = np.array([
	[1  + 5*1j, 1 + 6*1j, 1 + 3*1j],
    [-5 + 3*1j, 2 + 2*1j, 4 + 9*1j],
    [1  + 2*1j, 5 + 7*1j, 3 + 4*1j],
    [4  + 3*1j, 3 + 1*1j, 8 + 6*1j],
    [6  - 1*1j, 4 + 2*1j, 2 + 9*1j]], np.complex128, order = 'C')
B = np.array([
	[1  + 5*1j, 1 + 6*1j, 1 + 3*1j, 2 + 6*1j],
    [-5 + 3*1j, 2 + 2*1j, 4 + 9*1j, 5 + 7*1j],
    [1  + 2*1j, 5 + 7*1j, 3 + 4*1j, 4 + 8*1j]], np.complex128, order = 'C')

# NumPy matmul
Cnp = np.matmul(A, B)
print('NumPy: \n', Cnp)

# Complex matmul from pyLOM:
C = pyLOM.math.matmul(A, B)
print('pyLOM: \n', C)

print('Error: \n', C - Cnp)


pyLOM.cr_info()