import numpy as np
from scipy.linalg import ldl
import pyLOM

#Test matrix
A = np.array([
	[1 + 1*1j, 1 + 6*1j, 1 + 2*1j],
    [-5 + 3*1j, 2 + 3*1j, 4 + 1*1j],
    [1 + 4*1j, 5 + 7*1j, 3 + 8*1j]])
B = np.matmul(A, np.transpose(np.conjugate(A)))
print('Matrix to factorize: \n', B)

#Cholesky decomposition from NumPy
Bchol = np.linalg.cholesky(B)
print('Cholesky NumPy: \n', Bchol)

#Cholesky decomposition from pyLOM:
BpyLOM = pyLOM.math.cholesky(B)
print('Cholesky pyLOM: \n', BpyLOM)
