#!/usr/bin/env python
#
# Example of SVD utility so that it returns
# the same output as MATLAB.
#
# Last revision: 09/07/2021
from __future__ import print_function, division

import numpy as np
import pyLOM


## Code in serial test
# Define the matrix and run SVD
if pyLOM.utils.is_rank_or_serial(0):
	# Define matrix A 4x2
	A = np.array([[1,2],[3,4],[5,6],[7,8]],dtype=np.double,order='C')

	# Run SVD from numpy
	pyLOM.cr_start('SVD numpy',0)
	U, S, V = np.linalg.svd(A,full_matrices=False)
	pyLOM.cr_stop('SVD numpy',0)
	print('Numpy:')
	print(U.shape,U)
	print(S.shape,S)
	print(V.shape,V)

	# Run SVD from pyLOM - serial algorithm
	U, S, V = pyLOM.math.svd(A)
	print('pyLOM SVD:')
	print(U.shape,U)
	print(S.shape,S)
	print(V.shape,V)
else:
	A = None


## TSQR SVD also runs in parallel
# Scatter A among the processors
Ai = pyLOM.utils.mpi_scatter(A,root=0,do_split=True)

# Run TSQR SVD in parallel if needed
Ui, S, V = pyLOM.math.tsqr_svd(Ai)

# Gather Ui to processor 0
U = pyLOM.utils.mpi_gather(Ui,root=0)

if pyLOM.utils.is_rank_or_serial(0):
	print('pyLOM TSQR SVD:')
	print(U.shape,U)
	print(S.shape,S)
	print(V.shape,V)


pyLOM.cr_info()