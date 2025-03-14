#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
#
# Last rev: 07/02/2025
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp


def delete_snapshot(X, snap, axis=1):
	"""
	Remove snapshot from training data.
	"""
	p = cp if type(X) is cp.ndarray else np
	return p.delete(X, snap, axis=axis)


def set_random_elements_to_zero(vector, percentage):
	"""
	Randomly sets a specified percentage of elements in a vector to zero.

	Args:
		vector (np.ndarray): Input vector.
		percentage (float): Percentage of elements to set to zero.

	Returns:
		np.ndarray: Modified vector with zeros.
	"""
	p = cp if type(vector) is cp.ndarray else np
	modified_vector = vector.copy()
	num_zeros = int(len(vector) * percentage / 100)
	indices = p.random.choice(len(vector), num_zeros, replace=False)
	modified_vector[indices] = 0
	return modified_vector
