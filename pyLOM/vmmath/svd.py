#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - SVD.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import time, numpy as np

from ..utils.gpu import cp
from .maths      import matmul
from .qr         import tsqr, randomized_qr
from ..utils     import cr_nvtx as cr


@cr('math.svd')
def svd(A,method='gesdd'):
	'''
	Single value decomposition (SVD) using numpy.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.linalg.svd(A,full_matrices=False)

@cr('math.tsqr_svd')
def tsqr_svd(Ai):
	'''
	Single value decomposition (SVD) using TSQR algorithm from
	J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
	and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
	vol. 34, no. 1, pp. A206–A239, Jan. 2012,

	doi: 10.1137/080731992.

	Ai(m,n)  data matrix dispersed on each processor.

	Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
	S(n)     singular values.
	VT(n,n)  right singular vectors (transposed).
	'''
	# QR factorization on A
	Qi,R = tsqr(Ai)

	# Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	# At this point we have R and Qi scattered on the processors
	# Call SVD routine
	Ur, S, V = svd(R)
	# Compute Ui = Qi x Ur
	Ui = matmul(Qi, Ur)
	return Ui, S, V

@cr('math.randomized_svd')
def randomized_svd(Ai, r, q, seed=-1):
	'''
	Ai(m,n)  data matrix dispersed on each processor.
	r        target number of modes

	Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
	S(n)     singular values.
	VT(n,n)  right singular vectors (transposed).
	'''
	seed = int(time.time()) if seed < 0 else seed
	np.random.seed(seed=seed)

	Qi, B    = randomized_qr(Ai,r,q,seed=seed)
	Ur, S, V = svd(B)
	
	# Compute Ui = Qi x Ur
	Ui = matmul(Qi, Ur)

	return Ui, S, V