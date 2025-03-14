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
from .maths      import matmul, matmulp
from ..utils     import cr_nvtx as cr, MPI_RANK, MPI_SIZE, mpi_send, mpi_recv


def next_power_of_2(n):
	'''
	Find the next power of 2 of n
	'''
	p = 1
	if (n and not(n & (n - 1))):
		return n
	while (p < n): p <<= 1
	return p


@cr('math.qr')
def qr(A):
	'''
	QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.linalg.qr(A)

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

@cr('math.tsqr')
def tsqr(Ai):
	'''
	Parallel QR factorization of a real array using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	p = cp if type(Ai) is cp.ndarray else np
	_, n = Ai.shape
	# Algorithm 1 from Demmel et al (2012)
	# 1: QR Factorization on Ai to obtain Q1i and Ri
	Q1i, R    = qr(Ai)
	nextPower = next_power_of_2(MPI_SIZE)
	nlevels   = int(np.log2(nextPower))
	QW        = p.eye(n, dtype = Ai.dtype)
	C         = p.zeros((2*n, n), Ai.dtype)
	Q2l       = p.zeros((2*n*nlevels, n), Ai.dtype)
	blevel    = 1
	for ilevel in range(nlevels):
		# Store R in the upper part of the C matrix
		C[:n, :] = R
		# Decide who sends and who recieves, use R as buffer
		prank = MPI_RANK ^ blevel
		if MPI_RANK & blevel:
			if prank < MPI_SIZE:
				mpi_send(R, prank)
		else:
			if prank < MPI_SIZE:
				R = mpi_recv(source = prank)
				# Store R in the lower part of the C matrix
				C[n:, :] = R
				# 2: QR from the C matrix, reuse C and R
				Q2i, R = qr(C)
				# Store Q2i from this level
				Q2l[2*n*ilevel:2*n*ilevel+2*n, :] = Q2i
		blevel <<= 1
	# At this point R is correct on processor 0
	# Broadcast R and its part of the Q matrix
	if MPI_SIZE > 1:
		blevel = 1 << (nlevels - 1)
		mask   = blevel - 1
	for ilevel in reversed(range(nlevels)):
		if MPI_RANK & mask == 0:
			# Obtain Q2i for this level - use C as buffer
			C = Q2l[2*n*ilevel:2*n*ilevel+2*n, :]
			# Multiply by QW either set to identity or allocated to a value
			# Store into Q2i
			Q2i = matmul(C, QW)
			# Communications scheme
			prank = MPI_RANK^blevel
			if MPI_RANK & blevel:
				if prank < MPI_SIZE:
					C = mpi_recv(source = prank)
					# Recover R from the upper part of C and QW from the lower part
					R  = C[:n, :]
					QW = C[n:, :]
			else:
				if prank < MPI_SIZE:
					# Set up C matrix for sending
					# Store R in the upper part and Q2i on the lower part
					# Store Q2i of this rank to QW
					C[:n, :] = R
					C[n:, :] = Q2i[n:, :]
					QW       = Q2i[:n, :]
					mpi_send(C, prank)
		blevel >>= 1
		mask   >>= 1
	# Multiply Q1i and QW to obtain Qi
	Qi = matmul(Q1i, QW)
	return Qi,R

@cr('math.randomized_qr')
def randomized_qr(Ai, r, q, seed=-1):
	'''
	Ai(m,n)  data matrix dispersed on each processor.
	r        target number of modes

	Qi(m,r)  
	B (r,n) 
	'''
	p = cp if type(Ai) is cp.ndarray else np
	_, n  = Ai.shape
	seed = int(time.time()) if seed < 0 else seed
	p.random.seed(seed=seed)
	omega = p.random.rand(n, r).astype(Ai.dtype)
	Yi = matmul(Ai,omega)
	# QR factorization on A
	for j in range(q):
		Qi,_ = tsqr(Yi)
		Q2i  = matmulp(Ai.T,Qi)
		Yi   = matmul(Ai,Q2i)

	Qi,_ = tsqr(Yi)
	B    = matmulp(Qi.T,Ai)

	return Qi, B

@cr('math.init_qr_streaming')
def init_qr_streaming(Ai, r, q, seed=None):
	'''
	Ai(m,n)  data matrix dispersed on each processor.
	r        target number of modes

	Qi(m,r)  
	B (r,n) 
	'''
	p = cp if type(Ai) is cp.ndarray else np
	_, n  = Ai.shape
	seed = int(time.time()) if seed == None else seed
	p.random.seed(seed=seed)
	omega = p.random.rand(n, r).astype(Ai.dtype)
	Yi    = matmul(Ai,omega)
	# QR factorization on A
	for j in range(q):
		Qi,_ = tsqr(Yi)
		Q2i  = matmulp(Ai.T,Qi)
		Yi   = matmul(Ai,Q2i)

	Qi,_ = tsqr(Yi)
	B    = matmulp(Qi.T,Ai)

	return Qi, B, Yi

@cr('math.qr_iteration')
def update_qr_streaming(Ai, Q1, B1, Yo, r, q):
	'''
	Ai(m,n)  data matrix dispersed on each processor.
	r        target number of modes

	Qi(m,r)  
	B (r,n) 
	'''
	p = cp if type(Ai) is cp.ndarray else np
	_, n  = Ai.shape
	omega = np.random.rand(n, r).astype(Ai.dtype)
	Yn    = matmul(Ai,omega)
	for jj in range(q):
		Qpi,_ = tsqr(Yn)
		O2    = matmulp(Ai.T, Qpi)
		Yn    = matmul(Ai,O2)
	Yo   += Yn
	Q2,_  = tsqr(Yo)
	Q2Q1  = matmulp(Q2.T, Q1)
	B2o   = matmul(Q2Q1, B1)
	B2n   = matmulp (Q2.T, Ai)
	B2    = p.hstack((B2o, B2n))

	return Q2, B2, Yo

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