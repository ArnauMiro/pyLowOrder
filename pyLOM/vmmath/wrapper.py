#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np, scipy, nfft

from ..utils.cr     import cr_start, cr_stop
from ..utils.parall import MPI_RANK, mpi_gather, mpi_reduce
from ..utils.errors import raiseError


## Python functions
def transpose(A):
	'''
	Transposed of matrix A
	'''
	cr_start('math.transpose',0)
	At = np.transpose(A)
	cr_stop('math.transpose',0)
	return At

def vector_norm(v,start=0):
	'''
	L2 norm of a vector
	'''
	cr_start('math.vector_norm',0)
	norm = np.linalg.norm(v[start:],2)
	cr_stop('math.vector_norm',0)
	return norm

def matmul(A,B):
	'''
	Matrix multiplication C = A x B
	'''
	cr_start('math.matmul',0)
	C = np.matmul(A,B)
	cr_stop('math.matmul',0)
	return C

def vecmat(v,A):
	'''
	Vector times a matrix C = v x A
	'''
	cr_start('math.vecmat',0)
	C = np.zeros_like(A)
	for ii in range(v.shape[0]):
		C[ii,:] = v[ii]*A[ii,:]
	cr_stop('math.vecmat',0)
	return C

def diag(A):
	'''
	If A is a matrix it returns its diagonal, if its a vector it returns
	a diagonal matrix with A in its diagonal
	'''
	cr_start('math.diag',0)
	B = np.diag(A)
	cr_stop('math.diag',0)
	return B

def eigen(A):
	'''
	Eigenvalues and eigenvectors using numpy.
		real(n)   are the real eigenvalues.
		imag(n)   are the imaginary eigenvalues.
		vecs(n,n) are the right eigenvectors.
	'''
	cr_start('math.eigen',0)
	w,vecs = np.linalg.eig(A)
	real   = np.real(w)
	imag   = np.imag(w)
	cr_stop('math.eigen',0)
	return real,imag,vecs

def build_complex_eigenvectors(vecs, imag):
	'''
	Reconstruction of the right eigenvectors in complex format
	'''
	cr_start('math.build_complex_eigenvectors', 0)
	wComplex = np.zeros(vecs.shape, dtype = 'complex_')
	ivec = 0
	while ivec < vecs.shape[1] - 1:
		if imag[ivec] > np.finfo(np.double).eps:
			wComplex[:, ivec]     = vecs[:, ivec] + vecs[:, ivec + 1]*1j
			wComplex[:, ivec + 1] = vecs[:, ivec] - vecs[:, ivec + 1]*1j
			ivec += 2
		else:
			wComplex[:, ivec] = vecs[:, ivec] + 0*1j
			ivec = ivec + 1
	cr_stop('math.build_complex_eigenvectors', 0)
	return wComplex

def polar(real, imag):
	'''
	Present a complex number in its polar form given its real and imaginary part
	'''
	cr_start('math.polar', 0)
	mod = np.sqrt(real*real + imag*imag)
	arg = np.arctan2(imag, real)
	cr_stop('math.polar', 0)
	return mod, arg

def temporal_mean(X):
	'''
	Temporal mean of matrix X(m,n) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cr_start('math.temporal_mean',0)
	out = np.mean(X,axis=1)
	cr_stop('math.temporal_mean',0)
	return out

def subtract_mean(X,X_mean):
	'''
	Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
	and n is the number of snapshots.
	'''
	cr_start('math.subtract_mean',0)
	out = X - np.tile(X_mean,(X.shape[1],1)).T
	cr_stop('math.subtract_mean',0)
	return out

def svd(A):
	'''
	Single value decomposition (SVD) using numpy.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cr_start('math.svd',0)
	U, S, V = np.linalg.svd(A,full_matrices=False)
	cr_stop('math.svd',0)
	return U,S,V

def tsqr_svd(A):
	'''
	Single value decomposition (SVD) using Lapack.
		U(m,n)   are the POD modes.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	'''
	cr_start('math.tsqr_svd',0)
	# Algorithm 1 from Sayadi and Schmid (2016) - Q and R matrices
	# QR factorization on A
	Q1, R1 = np.linalg.qr(A)
	# Gather all Rs into Rp
	Rp = mpi_gather(R1,all=True)
	# QR factorization on Rp
	Q2, R = np.linalg.qr(Rp)
	# Compute Q = Q1 x Q2
	Q = np.matmul(Q1,Q2[A.shape[1]*MPI_RANK:A.shape[1]*(MPI_RANK+1),:])
	# At this point we have R and Qi scattered on the processors
	# Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	# Call SVD routine
	Ur, S, V = np.linalg.svd(R)
	# Compute U = Q x Ur
	U = np.matmul(Q,Ur)
	cr_stop('math.tsqr_svd',0)
	return U,S,V

def fft(t,y,equispaced=True):
	'''
	Compute the PSD of a signal y.
	'''
	cr_start('math.fft',0)
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
	ps = np.real(yf*np.conj(yf))/y.shape[0] # np.abs(yf)/y.shape[0]
	cr_stop('math.fft',0)
	return f, ps

def RMSE(A,B):
	'''
	Compute RMSE between X_POD and X
	'''
	cr_start('math.RMSE',0)
	diff  = (A-B)
	sum1g = mpi_reduce(np.sum(diff*diff),op='sum',all=True)
	sum2g = mpi_reduce(np.sum(A*A),op='sum',all=True)
	rmse  = np.sqrt(sum1g/sum2g)
	cr_stop('math.RMSE',0)
	return rmse

def vandermonde(real, imag, shape0, shape1):
	'''
	Builds a Vandermonde matrix of (shape0 x shape 1) with the real and imaginary parts of the eigenvalues
	'''
	cr.start('math.vandermonde', 0)
	mod, arg = polar(real, imag)
	Vand  = np.zeros((shape0, shape1), dtype = 'complex_')
	for icol in range(shape1):
		VandModulus   = mod**icol
		VandArg       = arg*icol
		Vand[:, icol] = mod*np.cos(arg) + mod*np.sin(arg)*1j
	cr_stop('math.vandermonde', 0)
	return Vand
