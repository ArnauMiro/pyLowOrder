#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Linear operator module
#
# Last rev: 31/08/2026
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp
from ..vmmath    import vecmat, matmul, tsqr_svd, transpose, matmulp, remove_rows, temporal_mean, subtract_mean, svd, dagger
# NOTE: truncation is taken from vmmath's own truncation module; reaching up to
# ..POD for it created a circular import (vmmath -> POD -> pyLOM) when the
# pure-Python mirrors are used (compiled vmmath calls the C routine directly)
from .truncation  import compute_truncation_residual
from ..utils     import cr_nvtx as cr, cr_start, cr_stop

def linear_operator(Y, Z, r):
	r'''
	Reduced order version of the linear operator (Jacobian)

	Args:
		Y (ndarray): snapshot matrix at time t
		Z (ndarray): snapshot matrix at time t + dt
		r (int or float): maximum truncation residual
			- r > 0: target residual
			- r < 0: fraction of cumulative energy to retain

	Results:
		U (ndarray): left singular vectors of Y
		S (ndarray): singular values of Y
		VT (ndarray): right singular vectors of Y
		Atilde (ndarray): reduced-order linear operator (Jacobian)
	'''
	# compiled vmmath helpers assume C-contiguous input
	p = cp if type(Y) is cp.ndarray else np
	Y = p.ascontiguousarray(Y)
	Z = p.ascontiguousarray(Z)

	cr_start('DMD.SVD',0)
	U_aux, S, VT = tsqr_svd(Y)
	cr_stop('DMD.SVD',0)
	if Y.shape[0] > Z.shape[0]:
		U = remove_rows(U_aux, Z.shape[0])
	else:
		U = U_aux.copy()

	# Truncate according to residual
	# NOTE: inlined POD.truncate (which we cannot import from here); follows the
	# compiled convention: r > 1 -> number of modes, 0 < r < 1 -> residual
	cr_start('DMD.truncate', 0)
	nr = int(r) if r > 1 else compute_truncation_residual(S, r)
	U, S, VT = U[:,:nr], S[:nr], VT[:nr,:]
	cr_stop('DMD.truncate', 0)

	# Project A (Jacobian of the snapshots) into POD basis
	cr_start('DMD.linear_mapping',0)
	aux1   = matmulp(transpose(U), Z)
	aux2   = transpose(vecmat(1./S, VT))
	Atilde = matmul(aux1, aux2)
	cr_stop('DMD.linear_mapping',0)

	return U, S, VT, Atilde

def concatenate(X=[], remove_mean=False):
	r'''
	Joins a list of snapshot matrices for the purpose of computing the linear operator (Jacobian) from them

	Args:
		X (list of ndarray): list of snapshot matrices
		remove_mean (bool): whether to remove the temporal mean from each snapshot matrix

	Results:
		Y (ndarray): snapshot matrix at time t
		Z (ndarray): snapshot matrix at time t + dt
	'''
	# Create the matrices Y, Z
	p = cp if type(X[0]) is cp.ndarray else np
	dtype = X[0].dtype
	m = X[0].shape[0]
	n = np.sum([M.shape[1] - 1 for M in X])
	if m > n:
		Y = p.zeros((m, n), dtype=dtype)
	else:
		Y = p.zeros((n, n), dtype=dtype)
	Z = p.zeros((m, n), dtype=dtype)

	# Fill the matrices
	offset = 0
	for M in X:
		ni = M.shape[1]
		if remove_mean:
			cr_start('DMD.temporal_mean',0)
			# compiled vmmath helpers assume C-contiguous input
			M = p.ascontiguousarray(M)
			mean = temporal_mean(M)

			Y[:m,offset:offset + ni-1] = subtract_mean(p.ascontiguousarray(M[:,:-1]), mean)
			Z[:,offset:offset + ni-1] = subtract_mean(p.ascontiguousarray(M[:,1:]), mean)
			offset += (ni-1)
			cr_stop('DMD.temporal_mean',0)
		else:
			Y[:m,offset:offset + ni-1] = M[:,:-1]
			Z[:,offset:offset + ni-1] = M[:,1:]
			offset += (ni-1)

	return Y, Z

def separate(X, remove_mean=False):
	r'''
	Separates a snapshot matrix into two matrices for the purpose of computing the linear operator (Jacobian) from them

	Args:
		X (ndarray): snapshot matrix
		remove_mean (bool): whether to remove the temporal mean from the snapshot matrix

	Results:
		Y (ndarray): snapshot matrix at time t
		Z (ndarray): snapshot matrix at time t + dt
	'''
	# Create the matrices Y, Z
	p = cp if type(X) is cp.ndarray else np
	dtype = X[0].dtype
	m = X.shape[0]
	n = X.shape[1]
	if m > (n-1):
		Y = p.zeros((m, n-1), dtype=dtype)
	else:
		Y = p.zeros((n-1, n-1), dtype=dtype)
	Z = p.zeros((m, n-1), dtype=dtype)

	# Fill the matrices
	if remove_mean:
		cr_start('DMD.temporal_mean',0)
		# compiled vmmath helpers assume C-contiguous input
		X = p.ascontiguousarray(X)
		mean = temporal_mean(X)

		Y[:m,:] = subtract_mean(p.ascontiguousarray(X[:,:-1]), mean)
		Z[:,:] = subtract_mean(p.ascontiguousarray(X[:,1:]), mean)
		cr_stop('DMD.temporal_mean',0)
	else:
		Y[:m,:] = X[:,:-1].copy()
		Z[:,:] = X[:,1:].copy()

	return Y, Z

def resolvent(A, w):
	r'''
	Resolvent of a matrix A at a given frequency w

	Args:
		A (ndarray): the matrix for which to compute the resolvent
		w (int or float): the frequency at which to compute the resolvent

	Returns:
		U (ndarray): response modes
		S (ndarray): energy gains
		V (ndarray): forcing modes
	'''
	# H_inv = (w * I - A)
	p = cp if type(A) is cp.ndarray else np
	H_inv = p.eye(A.shape[0]) * w - A 
	V_flip, S_inv, UT_flip = svd(H_inv)
	U_flip = dagger(UT_flip)
	S  = 1 / S_inv[::-1]
	U  = U_flip[:, ::-1]
	V  = V_flip[:, ::-1]
	return U, S, V

def flip_columns(A):
	r'''
	Flips the columns of a matrix A in a reverse order

	Args:
		A (ndarray): the matrix whose columns to flip

	Returns:
		B (ndarray): the matrix with flipped columns
	'''
	p = cp if type(A) is cp.ndarray else np
	B = p.fliplr(A)

	return B