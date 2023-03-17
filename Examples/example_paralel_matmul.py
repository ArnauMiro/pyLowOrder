#!/usr/bin/env python
#
# Example how to perform parallel matmul.
#
# Last revision: 01/02/2023
from __future__ import print_function, division

import numpy as np
import pyLOM


# Test matrix
A = np.array([
	[1,  1, 1, 2],
    [-5, 2, 4, 8]], order = 'C', dtype = np.double)
B = np.array([
	[3, 2, 9, 4],
    [8, 6, 1, 7]], order = 'C', dtype = np.double)

Ai = pyLOM.utils.mpi_scatter(A, root = 0, do_split = True)
Ai = np.transpose(Ai)
Bi = pyLOM.utils.mpi_scatter(B, root = 0, do_split = True)

# NumPy matmul
Caux = np.matmul(Ai, Bi)
Cnp  = pyLOM.utils.mpi_reduce(Caux, root = 0, op = 'sum', all = True)

pyLOM.pprint(0,'NumPy: \n', Cnp)

# Paralel matmul from pyLOM:
C = pyLOM.math.matmul_paral(Ai, Bi)
pyLOM.pprint(0,'pyLOM: \n', C)

pyLOM.cr_info()