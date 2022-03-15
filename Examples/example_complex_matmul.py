import numpy as np
from scipy.linalg import ldl
import pyLOM

#Test matrix
A = np.array([[1 + 0*1j, 2 + 0*1j, 3 + 0*1j]])
B = np.array([
	[1 + 1*1j, 1 + 6*1j, 1 + 2*1j],
    [-5 + 3*1j, 2 + 3*1j, 4 + 1*1j],
    [1 + 4*1j, 5 + 7*1j, 3 + 8*1j]])
print(A.shape)
#NumPy matmul
Cnp = np.matmul(A, B)
print('NumPy: \n', Cnp)

#Cholesky decomposition from pyLOM:
C = pyLOM.math.complex_matmul(A, B)
print('pyLOM: \n', C)
