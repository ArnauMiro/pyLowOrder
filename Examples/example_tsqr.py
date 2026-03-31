#!/usr/bin/env python
#
# Example of SVD utility so that it returns
# the same output as MATLAB.
#
# Last revision: 31/03/2026
from __future__ import print_function, division

import numpy as np
import pyLOM


## Code in serial test
# Define the matrix and run QR
if pyLOM.utils.is_rank_or_serial(0):
	# Define matrix A 4x2
	A = np.array([[1,2],[3,4],[5,6],[7,8]],dtype=np.double,order='C')

	# Run QR from numpy
	pyLOM.cr_start('QR numpy',0)
	Q, R = np.linalg.qr(A)
	pyLOM.cr_stop('QR numpy',0)
	print('Numpy:')
	print(Q.shape,Q)
	print(R.shape,R)

	# Run SVD from pyLOM - serial algorithm
	# Semi-positive unique solution: a phase can be added to the numpy solution
	Q, R = pyLOM.math.qr(A)
	print('pyLOM SVD:')
	print(Q.shape,Q)
	print(R.shape,R)
else:
	A = None


## TSQR SVD also runs in parallel
# Scatter A among the processors
Ai = pyLOM.utils.mpi_scatter(A,root=0,do_split=True)

# Run TSQR in parallel if needed
Qi, R = pyLOM.math.tsqr(Ai)

# Gather Ui to processor 0
Q = pyLOM.utils.mpi_gather(Qi,root=0)

if pyLOM.utils.is_rank_or_serial(0):
	print('pyLOM TSQR:')
	print(Q.shape,Q)
	print(R.shape,R)


pyLOM.cr_info()