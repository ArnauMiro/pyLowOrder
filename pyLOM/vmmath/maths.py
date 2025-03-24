#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - vector/matrix math.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp
from ..utils     import cr_nvtx as cr, mpi_reduce, gpu_to_cpu


## Python functions
@cr('math.transpose')
def transpose(A:np.ndarray) -> np.ndarray:
	r'''
	Transposed of matrix A

	Args:
		A (np.ndarray): Matrix to be transposed
	
	Results
		np.ndarray: Transposed matrix
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.transpose(A)

@cr('math.vector_norm')
def vector_sum(v:np.ndarray,start:int=0) -> float:
	r'''
	Sum of a vector

	Args:
		v (np.ndarray): a vector
		start (int): position of the vector where to start the sum
	
	Result:
		float: sum of the vector
	'''
	p = cp if type(v) is cp.ndarray else np
	return p.sum(v[start:])

@cr('math.vector_norm')
def vector_norm(v:np.ndarray,start:int=0) -> float:
	r'''
	L2 norm of a vector
	
	Args:
		v (np.ndarray): a vector
		start (int): position of the vector where to start the norm
	
	Result:
		float: norm of the vector
	'''
	p = cp if type(v) is cp.ndarray else np
	return p.linalg.norm(v[start:],2)

@cr('math.vector_mean')
def vector_mean(v:np.ndarray,start:int=0) -> float:
	r'''
	Mean of a vector

	Args:
		v (np.ndarray): a vector
		start (int): position of the vector where to start the mean
	
	Result:
		float: mean of the vector
	'''
	p = cp if type(v) is cp.ndarray else np
	return p.mean(v[start:])

@cr('math.matmul')
def matmul(A:np.ndarray,B:np.ndarray) -> np.ndarray:
	r'''
	Matrix multiplication 
	C = A x B

	Args:
		A (np.ndarray): Matrix A (M,Q)
		B (np.ndarray): Matrix B (Q,N)
	
	Result:
		np.ndarray: Resulting matrix C (M,N)
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.matmul(A,B)

@cr('math.matmulp')
def matmulp(A:np.ndarray,B:np.ndarray) -> np.ndarray:
	r'''
	Matrix multiplication in parallel
	C = A x B 

	.. warning::
	A and B are distributed along the processors and C is the same for all of them

	Args:
		A (np.ndarray): Matrix A (M,Q)
		B (np.ndarray): Matrix B (Q,N)
	
	Result:
		np.ndarray: Resulting matrix C (M,N)
	'''
	p = cp if type(A) is cp.ndarray else np
	aux = p.matmul(A,B)
	return mpi_reduce(aux, root = 0, op = 'sum', all = True)

@cr('math.vecmat')
def vecmat(v:np.ndarray,A:np.ndarray) -> np.ndarray:
	r'''
	Vector times a matrix 
	C = v x A

	Args:
		v (np.ndarray): Vector v (M,)
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: Resulting matrix C (M,N)
	'''
	p = cp if type(v) is cp.ndarray else np
	C = p.zeros_like(A)
	for ii in range(v.shape[0]):
		C[ii,:] = v[ii]*A[ii,:]
	return C

@cr('math.argsort')
def argsort(v:np.ndarray) -> np.ndarray:
	r'''
	Returns the indices that sort a vector

	Args:
		v (np.ndarray): Vector v (M,)
	
	Result:
		np.ndarray: Indices that sort v (M,)
	'''
	p = cp if type(v) is cp.ndarray else np
	return p.argsort(v)

@cr('math.diag')
def diag(A:np.ndarray) -> np.ndarray:
	r'''
	If A is a matrix it returns its diagonal, if its a vector it returns
	a diagonal matrix with A in its diagonal

	Args:
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: Diagonal of A (M,)
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.diag(A)

@cr('math.eigen')
def eigen(A:np.ndarray) -> np.ndarray:
	r'''
	Eigenvalues and eigenvectors.

	.. warning::
	GPU implementation of this algoritm is still slow and
	thus will be executed purely on CPU level until
	cupy implements linalg.eig

	Args:
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: the real eigenvalues, real(M)
		np.ndarray: the imaginary eigenvalues, imag(M)
		np.ndarray: the right eigenvectors, vecs(M,M)
	'''
	A = gpu_to_cpu(A)
	w,vecs = np.linalg.eig(A)
	real   = np.real(w)
	imag   = np.imag(w)
	return real, imag, vecs

@cr('math.polar')
def polar(real:np.ndarray, imag:np.ndarray) -> np.ndarray:
	r'''
	Present a complex number in its polar form given its real and imaginary part

	Args:
		real (np.ndarray): the real component
		imag (np.ndarray): the imaginary component

	Result:
		np.ndarray: the modulus
		np.ndarray: the argument
	'''
	p = cp if type(real) is cp.ndarray else np
	mod = p.sqrt(real*real + imag*imag)
	arg = p.arctan2(imag, real)
	return mod, arg

@cr('math.vandermonde')
def vandermonde(real:np.ndarray, imag:np.ndarray, m:int, n:int) -> np.ndarray:
	r'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues

	Args:
		real (np.ndarray): the real component
		imag (np.ndarray): the imaginary component
		m (int): number of rows for the Vandermode matrix
		n (int): number of columns for the Vandermode matrix

	Result:
		np.ndarray: the Vandermonde matrix
	'''
	p = cp if type(real) is cp.ndarray else np
	dtype = p.complex128 if real.dtype is p.double else p.complex64
	Vand  = p.zeros((m, n), dtype=dtype)
	for icol in range(n):
		Vand[:, icol] = (real + imag*1j)**icol
	return Vand

@cr('math.vandermondeTime')
def vandermondeTime(real:np.ndarray, imag:np.ndarray, m:int, time:np.ndarray) -> np.ndarray:
	r'''
	Builds a Vandermonde matrix of (m x n) with the real and
	imaginary parts of the eigenvalues


	Args:
		real (np.ndarray): the real component
		imag (np.ndarray): the imaginary component
		m (int): number of rows for the Vandermode matrix
		time (np.ndarray): the time vector

	Result:
		np.ndarray: the Vandermonde matrix
	'''
	p = cp if type(real) is cp.ndarray else np
	dtype = p.complex128 if real.dtype is p.double else p.complex64
	n = time.shape[0]
	Vand  = p.zeros((m, n), dtype=dtype)
	for it, t in enumerate(time):
		Vand[:, it] = (real + imag*1j)**t
	return Vand

@cr('math.cholesky')
def cholesky(A:np.ndarray) -> np.ndarray:
	r'''
	Returns the Cholesky decompositon of A

	Args:
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: Cholesky factorization of A (M,N)
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.linalg.cholesky(A)

@cr('math.conj')
def conj(A:np.ndarray) -> np.ndarray:
	r'''
	Conjugates complex number A

	Args:
		A (np.ndarray): Vector, matrix or number A
	
	Result:
		np.ndarray: Conjugate of A
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.conj(A)

@cr('math.inv')
def inv(A:np.ndarray) -> np.ndarray:
	r'''
	Computes the inverse matrix of A

	Args:
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: Inverse of A (M,N)
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.linalg.inv(A)

@cr('math.flip')
def flip(A:np.ndarray) -> np.ndarray:
	r'''
	Returns the pointwise conjugate of A

	.. warning::
	This function is not implemented in the
	compiled layer and will raise an error if used

	Args:
		A (np.ndarray): Matrix A (M,N)
	
	Result:
		np.ndarray: Flipped version of A (M,N)
	'''
	p = cp if type(A) is cp.ndarray else np
	return p.flip(A)