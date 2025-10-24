#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
#
# Last rev: 29/04/2025

from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp


def delete_snapshot(X, snap, axis=1):
	"""
	Remove one or more snapshots from a snapshot matrix.

	Args:
		X (np.ndarray or cp.ndarray): Input snapshot matrix of shape (n_features, n_samples).
		snap (int or sequence of int): Index or indices of snapshot columns to remove.
		axis (int, optional): Axis along which to delete snapshots (default is 1).

	Returns:
		np.ndarray or cp.ndarray: Snapshot matrix with specified columns removed.
	"""
	p = cp if type(X) is cp.ndarray else np
	return p.delete(X, snap, axis=axis)


def set_random_elements_to_zero(vector, percentage):
	"""
	Randomly zero out a specified percentage of elements in a vector.

	Args:
		vector (np.ndarray or cp.ndarray): 1D input array whose elements will be modified.
		percentage (float): Percentage (0 to 100) of elements to set to zero.

	Returns:
		np.ndarray or cp.ndarray: Copy of the input vector with the specified fraction of elements set to zero.
	"""
	p = cp if type(vector) is cp.ndarray else np
	modified_vector = vector.copy()
	num_zeros = int(len(vector) * percentage / 100)
	indices = p.random.choice(len(vector), num_zeros, replace=False)
	modified_vector[indices] = 0
	return modified_vector
