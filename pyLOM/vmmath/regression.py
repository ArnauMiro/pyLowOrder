#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - regression.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp
from .maths      import transpose, matmul, inv
from ..utils     import cr_nvtx as cr


# This equation could be moved to the vmath module
@cr('math.least_squares')
def least_squares(A,b):
	'''
	Least squares regression
	(A^T * A)^-1 * A^T * b
	'''
	A_t = transpose(A)
	normal_matrix = matmul(A_t, A)
	inv_normal_matrix = inv(normal_matrix)
	A_t_b = matmul(A_t, b)
	return matmul(inv_normal_matrix, A_t_b)
	
@cr('math.ridge_regresion')
def ridge_regresion(A,b,lam):
	'''
	Ridge regression
	'''
	p = cp if type(A) is cp.ndarray else np
	I = p.sqrt(lam)*p.eye(A.shape[1])
	augmented_A = p.vstack([A, I])
	augmented_b = p.hstack([b,p.zeros((I.shape[0],))])
	return least_squares(augmented_A,augmented_b)