#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for POD.
#
# Last rev: 09/07/2021
from __future__ import print_function

from ..vmmath       import vecmat, matmul, temporal_mean, subtract_mean, tsqr_svd, randomized_svd, compute_truncation_residual
from ..utils.cr     import cr_nvtx as cr, cr_start, cr_stop


## POD run method
@cr('POD.run')
def run(X,remove_mean=True, randomized=False, r=1, q=3, seed=-1):
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
	if remove_mean:
		cr_start('POD.temporal_mean',0)
		# Compute temporal mean
		X_mean = temporal_mean(X)
		# Compute substract temporal mean
		Y = subtract_mean(X,X_mean)
		cr_stop('POD.temporal_mean',0)
	else:
		Y = X.copy()
	# Compute SVD
	cr_start('POD.SVD',0)
	U,S,V = tsqr_svd(Y) if not randomized else randomized_svd(Y,r,q,seed=seed)
	cr_stop('POD.SVD',0)
	# Return
	return U,S,V


## POD truncate method
@cr('POD.truncate')
def truncate(U,S,V,r=1e-8):
	'''
	Truncate POD matrices (U, S, V) given a residual, number of modes or cumulative energy r.

	Inputs:
		- U(m,n)  are the POD modes.
		- S(n)    are the singular values.
		- V(n,n)  are the right singular vectors.
		- r       target residual, number of modes, or cumulative energy threshold.
					* If r >= 1, it is treated as the number of modes.
					* If r < 1 and r > 0 it is treated as the residual target.
					* If r < 1 and r < 0 it is treated as the fraction of cumulative energy to retain.
					Note:  must be in (0,-1] and r = -1 is valid

	Returns:
		- U(m,N)  are the POD modes (truncated at N).
		- S(N)    are the singular values (truncated at N).
		- V(N,n)  are the right singular vectors (truncated at N).
	'''
	# Compute N using S
	N = int(r) if r >= 1 else compute_truncation_residual(S, r)
	
 	# Truncate
	Ur = U[:,:N]
	Sr = S[:N]
	Vr = V[:N,:]
	# Return
	return Ur, Sr, Vr


## POD reconstruct method
@cr('POD.reconstruct')
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
	# Compute X = U x S x VT
	return matmul(U,vecmat(S,V))
