#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# PCA general utilities.
#
# Last rev: 11/04/2025
import numpy as np ## Only for documentation

from scipy.stats import f

from ..utils.gpu import cp
from ..utils.cr  import cr_nvtx as cr


@cr('PCA.T2')
def T2score(P:np.ndarray, ncomp:int=1, confidence:float=0.8):
	r'''
	Summarize the scores over the ncomp and return the limit for the confidence itnerval according to its probabilistic distribution.

	Args:
		P (np.ndarray): scores from PCA.
		ncomp (int, optional): number of components to summarize the scores from (default: ``1``).
		confidence (float, optional): threshold confidence interval for clustering according to the T2 statistic. It is computed as in a F probability distribution (default: ``0.8``).

	Returns:
		[(np.ndarray), float]: T2 scores and clustering threshold.
	'''
	
	p   = cp if type(P) is cp.ndarray else np
	m,n = 1, P.shape[0]
	cov = 1/(n-1)*p.linalg.norm(P[:,0])**2
	T2  = p.zeros((P.shape[1],))
	for ii in range(P.shape[1]):
		T2[ii] = p.sum(P[ii,:ncomp]**2/cov)

	f_critical = f.ppf(confidence, dfn=m, dfd=n-m)
	T2_limit   =  (n-1)*m/(n-m)*f_critical

	return T2, T2_limit