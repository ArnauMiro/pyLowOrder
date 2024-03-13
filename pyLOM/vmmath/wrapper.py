#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np, scipy, nfft
from mpi4py import MPI

from ..utils.cr     import cr
from ..utils.parall import mpi_gather, mpi_reduce, pprint, mpi_send, mpi_recv, is_rank_or_serial
from ..utils.errors import raiseError
import h5py


## Python functions
@cr('math.transpose')
def transpose(A):
	'''
	Transposed of matrix A
	'''
	return np.transpose(A)

@cr('math.vector_norm')
def vector_norm(v,start=0):
	'''
	L2 norm of a vector
	'''
	return np.linalg.norm(v[start:],2)

@cr('math.matmul')
def matmul(A,B):
	'''
	Matrix multiplication C = A x B
	'''
	return np.matmul(A,B)

@cr('math.matmulp')
def matmulp(A,B):
	'''
	Matrix multiplication C = A x B where A and B are distributed along the processors and C is the same for all of them
	'''
	aux = np.matmul(A,B)
	return mpi_reduce(aux, root = 0, op = 'sum', all = True)

@cr('math.vecmat')
def vecmat(v,A):
	'''
	Vector times a matrix C = v x A
	'''
	C = np.zeros_like(A)
	for ii in range(v.shape[0]):
		C[ii,:] = v[ii]*A[ii,:]
	return C

@cr('math.argsort')
def argsort(v):
	'''
	Returns the indices that sort a vector
	'''
	return np.argsort(v)

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
	w,vecs = np.linalg.eig(A)
	real   = np.real(w)
	imag   = np.imag(w)
	return real,imag,vecs

@cr('math.polar')
def polar(real, imag):
	'''
	Present a complex number in its polar form given its real and imaginary part
	'''
	mod = np.sqrt(real*real + imag*imag)
	arg = np.arctan2(imag, real)
	return mod, arg

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

@cr('math.qr')
def qr(A):
	'''
	QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	return np.linalg.qr(A)

@cr('math.svd')
def svd(A,method='gesdd'):
	'''
	Single value decomposition (SVD) using numpy.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
#	return np.linalg.svd(A,lapack_driver=method,check_finite=False,full_matrices=False)
	return np.linalg.svd(A,full_matrices=False)

@cr('math.tsqr2')
def tsqr2(A):
	'''
	Parallel QR factorization using Lapack
		Q(m,n) is the Q matrix
		R(n,n) is the R matrix
	'''
	# Algorithm 1 from Sayadi and Schmid (2016) - Q and R matrices
	# QR factorization on A
	Q1i, R = qr(A)
	# Gather all Rs into Rp
	Rp = mpi_gather(R,all=True)
	# QR factorization on Rp
	Q2i, R = qr(Rp)
	# Compute Q = Q1 x Q2
	Q = matmul(Q1i,Q2i[A.shape[1]*MPI_RANK:A.shape[1]*(MPI_RANK+1),:])
	return Q,R

@cr('math.tsqr_svd2')
def tsqr_svd2(A):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	# Algorithm 1 from Sayadi and Schmid (2016) - Q and R matrices
	# QR factorization on A
	Q,R = tsqr2(A)

	# Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	# At this point we have R and Qi scattered on the processors
	# Call SVD routine
	Ur, S, V = svd(R)
	# Compute U = Q x Ur
	U = matmul(Q,Ur)
	return U,S,V

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
	#Recover rank and size
	MPI_COMM = MPI.COMM_WORLD      # Communications macro
	MPI_RANK = MPI_COMM.Get_rank() # Who are you? who? who?
	MPI_SIZE = MPI_COMM.Get_size() # Total number of processors used (workers)
	m, n = Ai.shape
	# Algorithm 1 from Demmel et al (2012)
	# 1: QR Factorization on Ai to obtain Q1i and Ri
	Q1i, R    = qr(Ai)
	nextPower = next_power_of_2(MPI_SIZE)
	nlevels   = int(np.log2(nextPower))
	QW        = np.eye(n, dtype = Ai.dtype)
	C         = np.zeros((2*n, n), Ai.dtype)
	Q2l       = np.zeros((2*n*nlevels, n), Ai.dtype)
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

@cr('math.fft')
def fft(t,y,equispaced=True):
	'''
	Compute the PSD of a signal y.
	'''
	if equispaced:
		ts = t[1] - t[0] # Sampling time
		# Compute sampling frequency
		f  = 1./ts/t.shape[0]*np.arange(t.shape[0],dtype=np.double)
		# Compute power spectra using fft
		yf = scipy.fft.fft(y)
	else:
		# Compute sampling frequency
		k_left = (t.shape[0]-1.)/2.
		f      = (np.arange(t.shape[0],dtype=np.double)-k_left)/t[-1]
		# Compute power spectra using fft
		x  = -0.5 + np.arange(t.shape[0],dtype=np.double)/t.shape[0]
		yf = nfft.nfft_adjoint(x,y,len(t))
	ps = np.real(yf*conj(yf))/y.shape[0] # np.abs(yf)/y.shape[0]
	return f, ps

@cr('math.RMSE')
def RMSE(A,B):
	'''
	Compute RMSE between X_POD and X
	'''
	diff  = (A-B)
	sum1g = mpi_reduce(np.sum(diff*diff),op='sum',all=True)
	sum2g = mpi_reduce(np.sum(A*A),op='sum',all=True)
	rmse  = np.sqrt(sum1g/sum2g)
	return rmse

@cr('math.vandermonde')
def vandermonde(real, imag, m, n):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues
	'''
	Vand  = np.zeros((m, n), dtype = 'complex_')
	for icol in range(n):
		Vand[:, icol] = (real + imag*1j)**icol
	return Vand

@cr('math.vandermondeTime')
def vandermondeTime(real, imag, m, time):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues
	'''
	n = time.shape[0]
	Vand  = np.zeros((m, n), dtype = 'complex_')
	for it, t in enumerate(time):
		Vand[:, it] = (real + imag*1j)**t
	return Vand

@cr('math.cholesky')
def cholesky(A):
	'''
	Returns the Cholesky decompositon of A
	'''
	return np.linalg.cholesky(A)

@cr('math.conj')
def conj(A):
	'''
	Conjugates complex number A
	'''
	return np.conj(A)

@cr('math.inv')
def inv(A):
	'''
	Computes the inverse matrix of A
	'''
	return np.linalg.inv(A)

@cr('math.flip')
def flip(A):
	'''
	Changes order of the vector
	'''
	return np.flip(A)

@cr('math.cellCenters')
def cellCenters(xyz,conec):
	'''
	Compute the cell centers given a list 
	of elements.
	'''
	xyz_cen = np.zeros((conec.shape[0],xyz.shape[1]),np.double)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c = conec[ielem,conec[ielem,:]>=0]
		xyz_cen[ielem,:] = np.mean(xyz[c,:],axis=0)
	return xyz_cen
