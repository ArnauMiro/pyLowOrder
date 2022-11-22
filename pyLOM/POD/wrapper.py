#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for POD.
#
# Last rev: 09/07/2021
from __future__ import print_function

import numpy as np

from ..vmmath       import vector_norm, vecmat, matmul, temporal_mean, subtract_mean, tsqr_svd
from ..utils.cr     import cr_start, cr_stop
from ..utils.errors import raiseError


## POD run method
def run(X,remove_mean=True):
	'''
	Run POD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,n_temp_snapshots]: data matrix
		- remove_mean:                     whether or not to remove the mean flow

	Returns:
		- U:  are the POD modes.
		- S:  are the singular values.
		- V:  are the right singular vectors.
	'''
	cr_start('POD.run',0)
	if remove_mean:
		# Compute temporal mean
		X_mean = temporal_mean(X)
		# Compute substract temporal mean
		Y = subtract_mean(X,X_mean)
	else:
		Y = X.copy()
	# Compute SVD
	U,S,V = tsqr_svd(Y)
	# Return
	cr_stop('POD.run',0)
	return U,S,V


## POD truncate method
def _compute_truncation_residual(S, r):
	'''
	'''
	N = 0
	normS = vector_norm(S,0)
	for ii in range(S.shape[0]):
		accumulative = vector_norm(S,ii)/normS
		if accumulative < r: break
		N += 1
	return nr

def truncate(U,S,V,r=1e-8):
	'''
	Truncate POD matrices (U,S,V) given a residual or number of modes r.

	Inputs:
		- U(m,n)  are the POD modes.
		- S(n)    are the singular values.
		- V(n,n)  are the right singular vectors.
		- r       target residual or number of modes (if it is greater than 1 is treated as number of modes, else is treated as residual. Default 1e-8)

	Returns:
		- U(m,N)  are the POD modes (truncated at N).
		- S(N)    are the singular values (truncated at N).
		- V(N,n)  are the right singular vectors (truncated at N).
	'''
	cr_start('POD.truncate',0)
	# Compute N using S
	N = int(r) if r >= 1 else _compute_truncation_residual(S, r)

	# Truncate
	Ur = U[:,:N]
	Sr = S[:N]
	Vr = V[:N,:]
	# Return
	cr_stop('POD.truncate',0)
	return Ur, Sr, Vr


## POD reconstruct method
def reconstruct(U,S,V):
	'''
	Reconstruct the flow given the POD decomposition matrices
	that can be possibly truncated.
	N is the truncated size
	n is the number of snapshots

	Inputs:
		- U(m,N)  are the POD modes.
		- S(N)    are the singular values.
		- V(N,n)  are the right singular vectors.

	Outputs
		- X(m,n)  is the reconstructed flow.
	'''
	cr_start('POD.reconstruct',0)
	# Compute X = U x S x VT
	X = matmul(U,vecmat(S,V))
	cr_stop('POD.reconstruct',0)
	return X
