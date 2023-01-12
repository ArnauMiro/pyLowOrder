#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for DMD.
#
# Last rev: 30/09/2021
from __future__ import print_function, division

cimport cython
cimport numpy as np

import numpy as np
from mpi4py  import MPI

from libc.stdlib   cimport malloc, free
from libc.string   cimport memcpy, memset
from libc.math     cimport sqrt, log, atan2
from mpi4py.libmpi cimport MPI_Comm
from mpi4py        cimport MPI

from ..utils.cr     import cr_start, cr_stop
from ..utils.errors import raiseError

cdef extern from "vector_matrix.h":
	cdef void   c_transpose           "transpose"(double *A, double *B, const int m, const int n)
	cdef double c_vector_norm         "vector_norm"(double *v, int start, int n)
	cdef void   c_matmul              "matmul"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_matmul_paral        "matmul_paral"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_matmul_complex      "matmul_complex"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k, char *TransA, char *TransB)
	cdef void   c_vecmat              "vecmat"(double *v, double *A, const int m, const int n)
	cdef void   c_vecmat_complex      "vecmat_complex"(np.complex128_t *v, np.complex128_t *A, const int m, const int n)
	cdef int    c_eigen               "eigen"(double *real, double *imag, np.complex128_t *vecs, double *A, const int m, const int n)
	cdef int    c_cholesky            "cholesky"(np.complex128_t *A, int N)
	cdef void   c_vandermonde         "vandermonde"(np.complex128_t *Vand, double *real, double *imag, int m, int n)
	cdef void   c_vandermonde_time    "vandermondeTime"(np.complex128_t *Vand, double *real, double *imag, int m, int n, double* t)
	cdef int    c_inverse             "inverse"(np.complex128_t *A, int N, char *UoL)
	cdef void   c_sort_complex_array  "sort_complex_array"(np.complex128_t *v, int *index, int n)
cdef extern from "averaging.h":
	cdef void c_temporal_mean "temporal_mean"(double *out, double *X, const int m, const int n)
	cdef void c_subtract_mean "subtract_mean"(double *out, double *X, double *X_mean, const int m, const int n)
cdef extern from "svd.h":
	cdef int c_tsqr_svd "tsqr_svd"(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm)
	cdef int c_svd      "svd"(double *U, double *S, double *VT, double *Y, const int m, const int n)
cdef extern from "truncation.h":
	cdef int  c_compute_truncation_residual "compute_truncation_residual"(double *S, double res, const int n)
	cdef void c_compute_truncation          "compute_truncation"(double *Ur, double *Sr, double *VTr, double *U, double *S, double *VT, const int m, const int n, const int N)

## DMD run method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def run(double[:,:] X, double r, int remove_mean=True):
	'''
	Run DMD analysis of a matrix X.

	Inputs:
		- X[ndims*nmesh,n_temp_snapshots]: data matrix
		- remove_mean:                     whether or not to remove the mean flow
		- r:                               maximum truncation residual

	Returns:
		- Phi:      DMD Modes
		- muReal:   Real part of the eigenvalues
		- muImag:   Imaginary part of the eigenvalues
		- b:        Amplitude of the DMD modes
		- Variables needed to reconstruct flow
	'''
	cr_start('DMD.run',0)
	# Variables
	cdef int m = X.shape[0], n = X.shape[1], mn = min(m,n-1), retval
	cdef double *X_mean
	cdef double *Y
	cdef int iaux, icol, irow
	cdef MPI.Comm MPI_COMM = MPI.COMM_WORLD
	#Output arrays:
	# Allocate memory
	Y  = <double*>malloc(m*n*sizeof(double))

	#Remove mean if required
	if remove_mean:
		X_mean = <double*>malloc(m*sizeof(double))
		# Compute temporal mean
		c_temporal_mean(X_mean,&X[0,0],m,n)
		# Compute substract temporal mean
		c_subtract_mean(Y,&X[0,0],X_mean,m,n)
		free(X_mean)
	else:
		memcpy(Y,&X[0,0],m*n*sizeof(double))

	#Get the first N-1 snapshots: Y1 = Y[:,:-1]
	cdef double *Y1
	cdef double *Y2
	Y1 = <double*>malloc(m*(n-1)*sizeof(double))
	Y2 = <double*>malloc(m*(n-1)*sizeof(double))
	for irow in range(m):
		for icol in range(n-1):
			Y1[irow*(n-1) + icol] = Y[irow*n + icol]
			Y2[irow*(n-1) + icol] = Y[irow*n + icol + 1]
	free(Y)

	# Compute SVD
	cdef double *U
	cdef double *S
	cdef double *V
	U  = <double*>malloc(m*mn*sizeof(double))
	S  = <double*>malloc(mn*sizeof(double))
	V  = <double*>malloc((n-1)*mn*sizeof(double))
	retval = c_tsqr_svd(U, S, V, Y1, m, mn, MPI_COMM.ob_mpi)
	if not retval == 0: raiseError('Problems computing SVD!')
	free(Y1)

	#Truncate
	cdef int nr
	cdef double *Ur
	cdef double *Sr
	cdef double *Vr

	nr = int(r) if r > 1 else c_compute_truncation_residual(S,r,n-1)
	Ur = <double*>malloc(m*nr*sizeof(double))
	Sr = <double*>malloc(nr*sizeof(double))
	Vr = <double*>malloc(nr*mn*sizeof(double))
	c_compute_truncation(Ur,Sr,Vr,U,S,V,m,n-1,nr)
	
	free(U)
	free(V)
	free(S)

	#Project Jacobian of the snapshots into the POD basis
	cdef double *aux1
	cdef double *aux2
	cdef double *aux3
	cdef double *Atilde
	cdef double *Urt
	aux1   = <double*>malloc(nr*(n-1)*sizeof(double))
	aux2   = <double*>malloc(nr*(n-1)*sizeof(double))
	aux3   = <double*>malloc(nr*sizeof(double))
	Atilde = <double*>malloc(nr*nr*sizeof(double))
	Urt    = <double*>malloc(nr*m*sizeof(double))
	c_transpose(Ur, Urt, m, nr)
	c_matmul_paral(aux1, Urt, Y2, nr, n-1, m)
	for icol in range(n-1):
		for irow in range(nr):
			aux2[icol*nr + irow] = Vr[irow*(n-1) + icol]/Sr[irow]
	c_matmul(Atilde, aux1, aux2, nr, nr, n-1)
	free(aux1)
	free(aux3)
	free(Urt)

	#Compute eigenmodes
	cdef double *auxmuReal
	cdef double *auxmuImag
	cdef np.complex128_t *w
	auxmuReal = <double*>malloc(nr*sizeof(double))
	auxmuImag = <double*>malloc(nr*sizeof(double))
	w         = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))
	retval = c_eigen(auxmuReal,auxmuImag,w,Atilde,nr,nr)
	free(Atilde)

	#Computation of DMD modes
	cdef np.complex128_t *auxPhi
	cdef np.complex128_t *aux1C
	cdef np.complex128_t *aux2C
	auxPhi = <np.complex128_t*>malloc(m*nr*sizeof(np.complex128_t))
	aux1C  = <np.complex128_t*>malloc(nr*sizeof(np.complex128_t))
	aux2C  = <np.complex128_t*>malloc(nr*sizeof(np.complex128_t))
	for iaux in range(m):
		for icol in range(nr):
			aux1C[icol] = 0 + 0*1j
			for irow in range(n-1):
				aux1C[icol] += Y2[iaux*(n-1) + irow]*aux2[irow*nr + icol]
		c_matmul_complex(aux2C, aux1C, w, 1, nr, nr, 'N', 'N')
		memcpy(&auxPhi[iaux*nr], aux2C, nr*sizeof(np.complex128_t))
	free(aux2)
	free(Y2)

	cdef double a
	cdef double b
	cdef double c
	cdef double d
	cdef double div
	for icol in range(nr):
		c = auxmuReal[icol]
		d = auxmuImag[icol]
		div = c*c + d*d
		for iaux in range(m):
			a = auxPhi[iaux + m*icol].real
			b = auxPhi[iaux + m*icol].imag
			auxPhi[iaux + m*icol].real = (a*c + b*d)/div
			auxPhi[iaux + m*icol].imag = (b*c - a*d)/div

	#Amplitudes according to: Jovanovic et. al. 2014 DOI: 10.1063
	cdef np.complex128_t *auxbJov
	cdef np.complex128_t *aux3C
	cdef np.complex128_t *Vand
	cdef np.complex128_t *P
	cdef np.complex128_t *Pinv
	cdef np.complex128_t *q

	auxbJov = <np.complex128_t*>malloc(nr*sizeof(np.complex128_t))
	aux3C   = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))
	aux4C   = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))
	Vand    = <np.complex128_t*>malloc((nr*(n-1))*sizeof(np.complex128_t))
	P       = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))
	Pinv    = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))
	q       = <np.complex128_t*>malloc(nr*sizeof(np.complex128_t))

	c_vandermonde(Vand, auxmuReal, auxmuImag, nr, n-1)
	c_matmul_complex(aux3C, w, w, nr, nr, nr, 'C', 'N')
	c_matmul_complex(aux4C, Vand, Vand, nr, nr, n-1, 'N', 'C')

	for irow in range(nr):
		for icol in range(nr): #Loop on the columns of the Vandermonde matrix
			P[irow*nr + icol]  = aux3C[irow*nr + icol].real*aux4C[irow*nr + icol].real
			P[irow*nr + icol] += -aux3C[irow*nr + icol].real*aux4C[irow*nr + icol].imag*1j 
			P[irow*nr + icol] += aux3C[irow*nr + icol].imag*aux4C[irow*nr + icol].real*1j
			P[irow*nr + icol] += aux3C[irow*nr + icol].imag*aux4C[irow*nr + icol].imag
	retval = c_cholesky(P, nr)
	if not retval == 0: raiseError('Problems computing Cholesky factorization!')

	for iaux in range(nr):
		for irow in range(nr):
			aux1C[irow] = 0 + 0*1j
			for icol in range(n-1):#casting Vr to a complex, at the same time, it is multipilied per S and Vand
				aux1C[irow] += Sr[irow]*Vr[irow*(n-1) + icol]*(Vand[iaux*(n-1) + icol].real+Vand[iaux*(n-1) + icol].imag*1j)
			aux2C[irow] = w[irow*nr + iaux]
		c_matmul_complex(&q[iaux], aux1C, aux2C, 1, 1, nr, 'N', 'N')

	memcpy(Pinv, P, nr*nr*sizeof(np.complex128_t))
	cdef int ii
	cdef int jj
	for ii in range(nr):
		q[ii] = q[ii].real - q[ii].imag*1j
		for jj in range(nr - ii):
			P[ii*nr + ii+jj]   = P[(ii+jj)*nr + ii].real - P[(ii+jj)*nr + ii].imag*1j
			P[(ii+jj)*nr + ii] = Pinv[ii*nr + ii+jj].real - Pinv[ii*nr + ii+jj].imag*1j

	retval = c_inverse(Pinv, nr, 'L')
	if not retval == 0: raiseError('Problems computing the Inverse!')

	c_matmul_complex(aux1C, Pinv, q, nr, 1, nr, 'N', 'N')

	retval = c_inverse(P, nr, 'U')
	if not retval == 0: raiseError('Problems computing the Inverse!')

	c_matmul_complex(auxbJov, P, aux1C, nr, 1, nr, 'N', 'N')

	# Free allocated arrays before reordering
	free(Ur)
	free(Sr)
	free(Vr)
	free(aux1C)
	free(aux2C)
	free(aux3C)
	free(aux4C)
	free(w)
	free(Vand)
	free(q)
	free(P)
	free(Pinv)

	#Order modes and eigenvalues according to its amplitude
	cdef int *auxOrd
	auxOrd = <int*>malloc(nr*sizeof(int))
	cdef np.ndarray[np.double_t,ndim=1] muReal = np.zeros((nr),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] muImag = np.zeros((nr),dtype=np.double)
	cdef np.ndarray[np.complex128_t,ndim=2] Phi = np.zeros((m,nr),order='C',dtype=np.complex128)
	cdef np.ndarray[np.complex128_t,ndim=1] bJov = np.zeros((nr,),dtype=np.complex128)
	
	c_sort_complex_array(auxbJov, auxOrd, n)
	for ii in range(nr):
		muReal[ii] = auxmuReal[auxOrd[nr-ii]]
		muImag[ii] = auxmuImag[auxOrd[nr-ii]]
		bJov[ii]   = auxbJov[auxOrd[nr-ii]]
		for jj in range(m):
			Phi[jj,ii]  = auxPhi[jj + m*auxOrd[nr-ii]]
	
	#Free the variables that had to be ordered
	free(auxmuReal)
	free(auxmuImag)
	free(auxbJov)
	free(auxPhi)

	#Ensure that all conjugate modes are in the same order
	cdef bint p = 0
	cdef double iimag
	for ii in range(nr):
		if p == 1:
			p = 0
			continue
		iimag = muImag[ii]
		if iimag < 0:
			muImag[ii]        =  muImag[ii+1]
			muImag[ii+1]      = -muImag[ii]
			bJov[ii].imag     =  bJov[ii+1].imag
			bJov[ii+1].imag   = -bJov[ii].imag
			for jj in range(m):
				Phi[jj,ii].imag   =  Phi[jj,ii+1].imag
				Phi[jj,ii+1].imag = -Phi[jj,ii+1].imag
			p = 1
			continue
		if iimag > 0:
			p = 1
			continue
	
	# Return
	cr_stop('DMD.run',0)

	return muReal, muImag, Phi, bJov


## DMD frequency damping
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def frequency_damping(double[:] real, double[:] imag, double dt):
	'''
	Computation of the damping ratio and the frequency of each mode
	'''
	cr_start('DMD.frequency_damping', 0)
	n = real.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] delta = np.zeros((n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] omega = np.zeros((n),dtype=np.double)
	cdef int ii
	cdef double mod
	cdef double arg
	for ii in range(n):
		mod       = sqrt(real[ii]*real[ii] + imag[ii]*imag[ii])
		delta[ii] = log(mod)/dt
		arg       = atan2(imag[ii], real[ii])
		omega[ii] = arg/dt
	cr_stop('DMD.frequency_damping', 0)
	return delta, omega

## Flow reconstruction
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def reconstruction_jovanovic(np.complex128_t[:,:] Phi, double[:] muReal, double[:] muImag, double[:] t, np.complex128_t[:] bJov):
	'''
	Computation of the reconstructed flow from the DMD computations
	'''
	cr_start('DMD.reconstruction_jovanovic', 0)
	cdef int ii
	cdef int m  = Phi.shape[0]
	cdef int n  = t.shape[0]
	cdef int nr = Phi.shape[1]
	cdef np.complex128_t *Vand
	cdef np.ndarray[np.complex128_t,ndim=2] Zdmd = np.zeros((m,n),order='C',dtype=np.complex128)

	Vand = <np.complex128_t*>malloc(nr*n*sizeof(np.complex128_t))

	c_vandermonde_time(Vand, &muReal[0], &muImag[0], nr, n, &t[0])
	c_vecmat_complex(&bJov[0], Vand, nr, n)
	c_matmul_complex(&Zdmd[0,0], &Phi[0,0], Vand, m, n, nr, 'N', 'N')
	
	free(Vand)

	cr_stop('DMD.reconstruction_jovanovic', 0)
	return Zdmd.real
