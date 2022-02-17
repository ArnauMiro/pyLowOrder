import numpy as np
from scipy.linalg import ldl

#Test matrix
A = np.array([
	[1, 0, 1],
    [0, 2, 0],
    [1, 0, 3]])
print('A: \n', A)

#Cholesky decomposition from NumPy
Achol = np.linalg.cholesky(A)
print('Cholesky: \n', Achol)

#LDL decomposition from scipy
Aldl, a, b = ldl(A)
print('LDL: \n', Aldl)
