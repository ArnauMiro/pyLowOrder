#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - vector/matrix math.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np, cupy as cp

from ..utils.gpu import cp
from ..utils     import cr_nvtx as cr, mpi_reduce


## Python functions
@cr('math.transpose')
def transpose(A):
	'''
	Transposed of matrix A
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.transpose(A)

@cr('math.vector_norm')
def vector_sum(v,start=0):
	'''
	Sum of a vector
	'''
	p = cp if type(v) is cp.ndarray else np
	return p.sum(v[start:])

@cr('math.vector_norm')
def vector_norm(v,start=0):
	'''
	L2 norm of a vector
	'''
	p = cp if type(v) is cp.ndarray else np
	return p.linalg.norm(v[start:],2)

@cr('math.vector_mean')
def vector_mean(v,start=0):
	'''
	Mean of a vector
	'''
	p = cp if type(v) is cp.ndarray else np
	return p.mean(v[start:])

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
	p = cp if type(A) is cp.ndarray else np
	aux = p.matmul(A,B)
	return mpi_reduce(aux, root = 0, op = 'sum', all = True)

@cr('math.vecmat')
def vecmat(v,A):
	'''
	Vector times a matrix C = v x A
	'''
	p = cp if type(v) is cp.ndarray else np
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
	p = cp if type(A) is cp.ndarray else np
	return p.diag(A)

@cr('math.eigen')
def eigen(A):
	'''
	Eigenvalues and eigenvectors using numpy.
		real(n)   are the real eigenvalues.
		imag(n)   are the imaginary eigenvalues.
		vecs(n,n) are the right eigenvectors.
	'''
	A = cp.asnumpy(A) if type(A) is cp.ndarray else A
	w,vecs = np.linalg.eig(A)
	real   = np.real(w)
	imag   = np.imag(w)
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

@cr('math.vandermonde')
def vandermonde(real, imag, m, n):
	'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues
	'''
	p = cp if type(real) is cp.ndarray else np
	dtype = p.complex128 if real.dtype is p.double else p.complex64
	Vand  = p.zeros((m, n), dtype=dtype)
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
	dtype = p.complex128 if real.dtype is p.double else p.complex64
	n = time.shape[0]
	Vand  = p.zeros((m, n), dtype=dtype)
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