import numpy as np
from scipy.linalg import ldl
import pyLOM
from pyLOM.utils.parall import mpi_scatter, mpi_reduce


#Test matrix
A = np.array([
	[1,  1, 1, 2],
    [-5, 2, 4, 8]], order = 'C', dtype = np.double)
B = np.array([
	[3, 2, 9, 4],
    [8, 6, 1, 7]], order = 'C', dtype = np.double)

Ai = mpi_scatter(A, root = 0, do_split = True)
Ai = np.transpose(Ai)
Bi = mpi_scatter(B, root = 0, do_split = True)

#NumPy matmul
Caux = np.matmul(Ai, Bi)
Cnp  = mpi_reduce(Caux, root = 0, op = 'sum', all = True)

pyLOM.pprint(0,'NumPy: \n', Cnp)

#Paralel matmul from pyLOM:
C = pyLOM.math.matmul_paral(Ai, Bi)
pyLOM.pprint(0,'pyLOM: \n', C)
