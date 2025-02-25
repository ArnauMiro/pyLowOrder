#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import time
import numpy as np, cupy as cp, scipy, nfft

from ..utils import cr
from ..utils import MPI_RANK, MPI_SIZE, mpi_reduce, mpi_send, mpi_recv


## Python functions
@cr('math.transpose')
def transpose(A):
	'''
	Transposed of matrix A
	'''
	return np.transpose(A)

@cr('math.vector_norm')
def vector_sum(v,start=0):
	'''
	Sum of a vector
	'''
	return np.sum(v[start:])

@cr('math.vector_norm')
def vector_norm(v,start=0):
	'''
	L2 norm of a vector
	'''
	p = cp if type(v) is cp.ndarray else np
	return p.linalg.norm(v[start:],2)

@cr('math.matmul')
def matmul(A,B):
	'''
	Matrix multiplication C = A x B
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.matmul(A,B)

@cr('math.matmulp')
def matmulp(A,B):
	'''
	Matrix multiplication C = A x B where A and B are distributed along the processors and C is the same for all of them
	'''
	aux = matmul(A,B)
	return mpi_reduce(aux, root = 0, op = 'sum', all = True)

@cr('math.vecmat')
def vecmat(v,A):
	'''
	Vector times a matrix C = v x A
	'''
	p = cp if type(A) is cp.ndarray else np
	C = p.zeros_like(A)
	for ii in range(v.shape[0]):
		C[ii,:] = v[ii]*A[ii,:]
	return C

@cr('math.argsort')
def argsort(v):
	'''
	Returns the indices that sort a vector
	'''
	p = cp if type(v) is cp.ndarray else np
	return p.argsort(v)

@cr('math.diag')
def diag(A):
	'''
	If A is a matrix it returns its diagonal, if its a vector it returns
	a diagonal matrix with A in its diagonal
	'''
	return np.diag(A)

@cr('math.eigen')
def eigen(A):
	'''
	Eigenvalues and eigenvectors using numpy.
		real(n)   are the real eigenvalues.
		imag(n)   are the imaginary eigenvalues.
		vecs(n,n) are the right eigenvectors.
	'''
	p = cp if type(A) is cp.ndarray else np
	w,vecs = p.linalg.eigh(A)
	real   = p.real(w)
	imag   = p.imag(w)
	return real,imag,vecs

@cr('math.polar')
def polar(real, imag):
	'''
	Present a complex number in its polar form given its real and imaginary part
	'''
	p = cp if type(real) is cp.ndarray else np
	mod = p.sqrt(real*real + imag*imag)
	arg = p.arctan2(imag, real)
	return mod, arg

@cr('math.temporal_mean')
def temporal_mean(X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	p = cp if type(X) is cp.ndarray else np
	return p.mean(X,axis=1)

@cr('math.subtract_mean')
def subtract_mean(X,X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	p = cp if type(X) is cp.ndarray else np
	return X - p.tile(X_mean,(X.shape[1],1)).T

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

def next_power_of_2(n):
	'''
	Find the next power of 2 of n
	'''
	p = 1
	if (n and not(n & (n - 1))):
		return n
	while (p < n): p <<= 1
	return p

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
	nlevels   = int(p.log2(nextPower))
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
	omega = p.random.rand(n, r).astype(Ai.dtype)
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
	p = cp if type(Ai) is cp.ndarray else np
	seed = int(time.time()) if seed < 0 else seed
	p.random.seed(seed=seed)

	Qi, B    = randomized_qr(Ai,r,q,seed=seed)
	Ur, S, V = svd(B)
	
	# Compute Ui = Qi x Ur
	Ui = matmul(Qi, Ur)

	return Ui, S, V

@cr('math.fft')
def fft(t,y,equispaced=True):
	'''
	Compute the PSD of a signal y.
	'''
	if equispaced:
		ts = t[1] - t[0] # Sampling time
		# Compute sampling frequency
		f  = 1./ts/t.shape[0]*np.arange(t.shape[0],dtype=y.dtype)
		# Compute power spectra using fft
		yf = scipy.fft.fft(y)
	else:
		# Compute sampling frequency
		k_left = (t.shape[0]-1.)/2.
		f      = (np.arange(t.shape[0],dtype=y.dtype)-k_left)/t[-1]
		# Compute power spectra using fft
		x  = -0.5 + np.arange(t.shape[0],dtype=y.dtype)/t.shape[0]
		yf = nfft.nfft_adjoint(x,y,len(t))
	ps = np.real(yf*conj(yf))/y.shape[0] # np.abs(yf)/y.shape[0]
	return f, ps

@cr('math.RMSE')
def RMSE(A,B):
	'''
	Compute RMSE between X_POD and X
	'''
	p = cp if type(A) is cp.ndarray else np
	diff  = (A-B)
	sum1g = mpi_reduce(p.sum(diff*diff),op='sum',all=True)
	sum2g = mpi_reduce(p.sum(A*A),op='sum',all=True)
	rmse  = p.sqrt(sum1g/sum2g)
	return rmse

@cr('math.energy')
def energy(original, rec):
	'''
	Compute reconstruction energy as in:
	Eivazi, H., Le Clainche, S., Hoyas, S., & Vinuesa, R. (2022). 
	Towards extraction of orthogonal and parsimonious non-linear modes from turbulent flows. 
	Expert Systems with Applications, 202, 117038.
	https://doi.org/10.1016
	'''
	# Compute local sums
	local_num = np.sum((original - rec) ** 2)
	local_den = np.sum(original ** 2)

	# Use Allreduce to compute global sums and make them available on all ranks
	global_num = mpi_reduce(local_num,op='sum',all=True)
	global_den = mpi_reduce(local_den,op='sum',all=True)

	# Compute Ek (this will be identical on all ranks)
	return 1 - global_num / global_den

@cr('math.vandermonde')
def vandermonde(real, imag, m, n):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues
	'''
	p = cp if type(real) is cp.ndarray else np
	Vand  = p.zeros((m, n), dtype = 'complex_')
	for icol in range(n):
		Vand[:, icol] = (real + imag*1j)**icol
	return Vand

@cr('math.vandermondeTime')
def vandermondeTime(real, imag, m, time):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues
	'''
	p = cp if type(real) is cp.ndarray else np
	n = time.shape[0]
	Vand  = p.zeros((m, n), dtype = 'complex_')
	for it, t in enumerate(time):
		Vand[:, it] = (real + imag*1j)**t
	return Vand

@cr('math.cholesky')
def cholesky(A):
	'''
	Returns the Cholesky decompositon of A
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.linalg.cholesky(A)

@cr('math.conj')
def conj(A):
	'''
	Conjugates complex number A
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.conj(A)

@cr('math.inv')
def inv(A):
	'''
	Computes the inverse matrix of A
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.linalg.inv(A)

@cr('math.flip')
def flip(A):
	'''
	Changes order of the vector
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.flip(A)

@cr('math.cellCenters')
def cellCenters(xyz,conec):
	'''
	Compute the cell centers given a list 
	of elements.
	'''
	p = cp if type(xyz) is cp.ndarray else np
	xyz_cen = p.zeros((conec.shape[0],xyz.shape[1]),xyz.dtype)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c = conec[ielem,conec[ielem,:]>=0]
		xyz_cen[ielem,:] = p.mean(xyz[c,:],axis=0)
	return xyz_cen

@cr('math.normals')
def normals(xyz,conec):
	p = cp if type(xyz) is cp.ndarray else np
	normals = p.zeros(((conec.shape[0],3)),xyz.dtype)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c     = conec[ielem,conec[ielem,:]>=0]
		xyzel =  xyz[c,:]
		# Compute centroid
		cen  = p.mean(xyzel,axis=0)
		# Compute normal
		for inod in range(len(c)):
			u = xyzel[inod]   - cen
			v = xyzel[inod-1] - cen
			normals[ielem,:] += 0.5*p.cross(u,v)
	return normals

@cr('math.euclidean_d')
def euclidean_d(X):
	'''
	Compute Euclidean distances between simulations.

	In:
		- X: NxM Data matrix with N points in the mesh for M simulations
	Returns:
		- D: MxM distance matrix 
	'''
	p = cp if type(X) is cp.ndarray else np
	# Extract dimensions
	_,M = X.shape
	# Initialize distance matrix
	D = p.zeros((M,M),X.dtype)
	for i in range(M):
		for j in range(i+1,M,1):
			# Local sum on the partition
			d2 = p.sum((X[:,i]-X[:,j])*(X[:,i]-X[:,j]))
			# Global sum over the partitions
			dG = p.sqrt(mpi_reduce(d2,all=True))
			# Fill output
			D[i,j] = dG
			D[j,i] = dG
	# Return the mdistance matrix
	return D
