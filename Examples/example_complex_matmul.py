import numpy as np
from scipy.linalg import ldl
import pyLOM

#Test matrix
A = np.array([
	[1  + 5*1j, 1 + 6*1j, 1 + 3*1j],
    [-5 + 3*1j, 2 + 2*1j, 4 + 9*1j],
    [1  + 2*1j, 5 + 7*1j, 3 + 4*1j],
    [4  + 3*1j, 3 + 1*1j, 8 + 6*1j],
    [6  - 1*1j, 4 + 2*1j, 2 + 9*1j]], order = 'C')
B = np.array([
	[1  + 5*1j, 1 + 6*1j, 1 + 3*1j, 2 + 6*1j],
    [-5 + 3*1j, 2 + 2*1j, 4 + 9*1j, 5 + 7*1j],
    [1  + 2*1j, 5 + 7*1j, 3 + 4*1j, 4 + 8*1j]], order = 'C')

#NumPy matmul
Cnp = np.matmul(A, B)
print('NumPy: \n', Cnp)

#Complex matmul from pyLOM:
C = pyLOM.math.complex_matmul(A, B)
print('pyLOM: \n', C)

print('Error: \n', C - Cnp)
