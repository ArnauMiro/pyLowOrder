import numpy as np
from scipy.linalg import ldl
import pyLOM

#Test matrix
#A = np.array([[1 + 0*1j, 2 + 0*1j, 3 + 0*1j]])
A = np.array([
	[1  + 0*1j, 1 + 0*1j, 1 + 0*1j],
    [-5 + 0*1j, 2 + 0*1j, 4 + 0*1j],
    [1  + 0*1j, 5 + 0*1j, 3 + 0*1j]], order = 'C')
B = A
B = np.array(B, order = 'C')
TransA = 'N'
TransB = 'C'
#NumPy matmul
Cnp = np.matmul(A, np.transpose(np.conj(B)))
print('NumPy: \n', Cnp)

#Cholesky decomposition from pyLOM:
C = pyLOM.math.complex_matmul(A, B, TransA, TransB)
print('pyLOM: \n', C)
