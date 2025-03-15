#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for POD.
#
# Last rev: 09/07/2021
from __future__ import print_function
import numpy as np ## Only for documentation

from ..vmmath       import vecmat, matmul, temporal_mean, subtract_mean, tsqr_svd, randomized_svd, compute_truncation_residual
from ..utils.cr     import cr_nvtx as cr, cr_start, cr_stop


## POD run method
@cr('POD.run')
def run(X,remove_mean:bool=True, randomized:bool=False, r:int=1, q:int=3, seed:int=-1):
	r'''
	Run POD analysis of a matrix.

	Args:
		X (np.array): data matrix of size [ndims*nmesh,n_temp_snapshots].
		remove_mean (bool, optional): whether or not to remove the mean flow (default: ``True``).
		randomized (bool, optional): whether to perform randomized POD or not (default: ``False``).
		r (int, optional): in case of performing randomized POD, how many modes do we want to recover. This option has no effect when randomized=False (default: ``1``).
		q (int, optional): in case of performing randomized POD, how many power iterations are performed. This option has no effect when randomized=False (default: ``3``).
		seed (int, optional): seed for reproducibility of randomized operations. This option has no effect when randomized=False (default: ``-1``).

	Returns:
		U (np.array): are the POD modes.
		S (np.array): are the singular values.
		V (np.array): are the right singular vectors.
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
def truncate(U:np.array,S:np.array,V:np.array,r:float=1e-8):
	r'''
	Truncate POD matrices (U, S, V) given a residual, number of modes or cumulative energy r.

	Args:
		U (np.array): of size (m,n), are the POD modes.
		S (np.array): of size (n), are the singular values.
		V (np.array): of size (n,n), are the right singular vectors.
		r (float, optional) target residual, number of modes, or cumulative energy threshold (default: ``1e-8``).
			* If r >= 1, it is treated as the number of modes.
			* If r < 1 and r > 0 it is treated as the residual target.
			* If r < 1 and r < 0 it is treated as the fraction of cumulative energy to retain.
			Note:  must be in (0,-1] and r = -1 is valid

	Returns:
		U (np.array): of size (m,N), are the POD modes (truncated at N).
		S (np.array): of size (N), are the singular values (truncated at N).
		V (np.array): of size (N,n), are the right singular vectors (truncated at N).
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
def reconstruct(U:np.array,S:np.array,V:np.array):
	r'''
	Reconstruct the flow given the POD decomposition matrices
	that can be possibly truncated.
	N is the truncated size
	n is the number of snapshots

	Args:
		U (np.array): of size (m,n), are the POD modes.
		S (np.array): of size (n), are the singular values.
		V (np.array): of size (n,n), are the right singular vectors.

	Returns:
		X (np.array): of size(m,n), is the reconstructed flow.
	'''
	# Compute X = U x S x VT
	return matmul(U,vecmat(S,V))
