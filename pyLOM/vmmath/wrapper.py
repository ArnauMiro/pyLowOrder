#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np, scipy

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
	out = X - X_mean
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

def fft(t,y):
	'''
	Compute the PSD of a signal y.
	'''
	cr_start('math.fft',0)
	ts = t[1] - t[0] # Sampling time
	# Compute sampling frequency
	f  = scipy.fft.fftfreq(y.size,ts)
	# Compute power spectra using fft 
	yf = scipy.fft.fft(y)
	ps = np.real(yf*np.conj(yf))/y.shape[0]
	# Rearrange the values
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