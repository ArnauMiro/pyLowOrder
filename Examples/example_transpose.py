import numpy as np
from scipy.linalg import ldl
import pyLOM

#Test matrix
A = np.array([
	[1, 6, 2, 9],
    [-5, 3, 4, -8],
    [4, 7, 8, 3]],dtype = np.double)
print('Matrix to transpose: \n', A)

#Cholesky decomposition from NumPy
Bnp = np.transpose(A)
print('Transpose NumPy: \n', Bnp)

#Cholesky decomposition from pyLOM:
BpyLOM = pyLOM.math.transpose(A)
print('Transpose pyLOM: \n', BpyLOM)
