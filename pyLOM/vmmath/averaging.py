#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - averaging.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.cr     import cr


## Python functions
@cr('math.temporal_mean')
def temporal_mean(X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	return np.mean(X,axis=1)

@cr('math.subtract_mean')
def subtract_mean(X,X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	return X - np.tile(X_mean,(X.shape[1],1)).T