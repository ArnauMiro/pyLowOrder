#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for PCA.
#
# Last rev: 11/04/2025
from __future__ import print_function
import numpy as np ## Only for documentation

from scipy.stats import f

from ..POD       import run as podrun
from ..vmmath    import vecmat
from ..utils.cr  import cr_nvtx as cr

@cr('PCA.run')
def run(X:np.ndarray, divide_variance:bool=True, randomized:bool=False, r:int=1, q:int=3, seed:int=-1):
	r'''
	Run PCA analysis of a matrix.

	Args:
		X (np.ndarray): data matrix of size [ndims*nmesh,n_temp_snapshots].
		divide_variance (bool, optional): whether or not to normalize the data with the variance. It is only effective when removing the mean (default: ``False``).
		randomized (bool, optional): whether to perform randomized PCA or not (default: ``False``).
		r (int, optional): in case of performing randomized PCA, how many modes do we want to recover. This option has no effect when randomized=False (default: ``1``).
		q (int, optional): in case of performing randomized PCA, how many power iterations are performed. This option has no effect when randomized=False (default: ``3``).
		seed (int, optional): seed for reproducibility of randomized operations. This option has no effect when randomized=False (default: ``-1``).

	Returns:
		[(np.ndarray), (np.ndarray)]: components and scores.
	'''

	U,S,V = podrun(X, remove_mean=True, divide_variance=divide_variance, randomized=randomized, r=r, q=q, seed=seed)

	return U, vecmat(S,V).T