import numpy as np
import pyLOM
from scipy.linalg import ldl

#Test matrix
A = np.array([
	[1.0000,0.5000,0.3333,0.2500],
    [0.5000,1.0000,0.6667,0.5000],
    [0.3333,0.6667,1.0000,0.7500],
    [0.2500,0.5000,0.7500,-1.000]])

#Cholesky decomposition from NumPy
try:
	print(np.linalg.cholesky(A))
except:
	print(ldl(A))
