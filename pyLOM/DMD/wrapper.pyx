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
from libc.complex  cimport creal, cimag
from mpi4py.libmpi cimport MPI_Comm
from mpi4py        cimport MPI

from ..utils.cr     import cr, cr_start, cr_stop
from ..utils.errors import raiseError

cdef extern from "vector_matrix.h":
	cdef void   c_transpose           "transpose"(double *A, double *B, const int m, const int n)
	cdef double c_vector_norm         "vector_norm"(double *v, int start, int n)
	cdef void   c_matmul              "matmul"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_matmulp             "matmulp"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_vecmat              "vecmat"(double *v, double *A, const int m, const int n)
	# Double complex precision
	cdef void   c_zmatmult            "zmatmult"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_zvecmat             "zvecmat"(np.complex128_t *v, np.complex128_t *A, const int m, const int n)
	cdef int    c_zinverse            "zinverse"(np.complex128_t *A, int N, char *UoL)
	cdef int    c_cholesky            "cholesky"(np.complex128_t *A, int N)
	cdef int    c_eigen               "eigen"(double *real, double *imag, np.complex128_t *vecs, double *A, const int m, const int n)
	cdef void   c_vandermonde         "vandermonde"(np.complex128_t *Vand, double *real, double *imag, int m, int n)
	cdef void   c_vandermonde_time    "vandermondeTime"(np.complex128_t *Vand, double *real, double *imag, int m, int n, double* t)
	cdef void   c_zsort               "zsort"(np.complex128_t *v, int *index, int n)
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
@cr('DMD.run')
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
		cr_start('DMD.temporal_mean',0)
		X_mean = <double*>malloc(m*sizeof(double))
		# Compute temporal mean
		c_temporal_mean(X_mean,&X[0,0],m,n)
		# Compute substract temporal mean
		c_subtract_mean(Y,&X[0,0],X_mean,m,n)
		free(X_mean)
		cr_stop('DMD.temporal_mean',0)
	else:
		memcpy(Y,&X[0,0],m*n*sizeof(double))

	#Get the first N-1 snapshots: Y1 = Y[:,:-1]
	cr_start('DMD.split_snapshots', 0)
	cdef double *Y1
	cdef double *Y2
	Y1 = <double*>malloc(m*(n-1)*sizeof(double))
	Y2 = <double*>malloc(m*(n-1)*sizeof(double))
	for irow in range(m):
		for icol in range(n-1):
			Y1[irow*(n-1) + icol] = Y[irow*n + icol]
			Y2[irow*(n-1) + icol] = Y[irow*n + icol + 1]
	free(Y)
	cr_stop('DMD.split_snapshots', 0)

	# Compute SVD
	cr_start('DMD.SVD',0)
	cdef double *U
	cdef double *S
	cdef double *V
	U  = <double*>malloc(m*mn*sizeof(double))
	S  = <double*>malloc(mn*sizeof(double))
	V  = <double*>malloc((n-1)*mn*sizeof(double))
	retval = c_tsqr_svd(U, S, V, Y1, m, mn, MPI_COMM.ob_mpi)
	cr_stop('DMD.SVD',0)
	if not retval == 0: raiseError('Problems computing SVD!')
	free(Y1)

	#Truncate
	cr_start('DMD.truncate',0)
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
	cr_stop('DMD.truncate',0)

	#Project Jacobian of the snapshots into the POD basis
	cr_start('DMD.linear_mapping',0)
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
	c_matmulp(aux1, Urt, Y2, nr, n-1, m)
	for icol in range(n-1):
		for irow in range(nr):
			aux2[icol*nr + irow] = Vr[irow*(n-1) + icol]/Sr[irow]
	c_matmul(Atilde, aux1, aux2, nr, nr, n-1)
	free(aux1)
	free(aux3)
	free(Urt)
	cr_stop('DMD.linear_mapping',0)

	#Compute eigenmodes
	
	cdef double *auxmuReal
	cdef double *auxmuImag
	cdef np.complex128_t *w
	auxmuReal = <double*>malloc(nr*sizeof(double))
	auxmuImag = <double*>malloc(nr*sizeof(double))
	w         = <np.complex128_t*>malloc(nr*nr*sizeof(np.complex128_t))
	cr_start('DMD.eigendecomposition',0)
	retval = c_eigen(auxmuReal,auxmuImag,w,Atilde,nr,nr)
	cr_stop('DMD.eigendecomposition',0)
	free(Atilde)

	#Computation of DMD modes
	cr_start('DMD.modes',0)
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
		c_zmatmult(aux2C, aux1C, w, 1, nr, nr, 'N', 'N')
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
			a = creal(auxPhi[iaux*nr + icol])
			b = cimag(auxPhi[iaux*nr + icol])
			auxPhi[iaux*nr + icol] = (a*c + b*d)/div + (b*c - a*d)/div*1j
	cr_stop('DMD.modes',0)

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

	cr_start('DMD.amplitudes', 0)
	c_vandermonde(Vand, auxmuReal, auxmuImag, nr, n-1)
	c_zmatmult(aux3C, w, w, nr, nr, nr, 'C', 'N')
	c_zmatmult(aux4C, Vand, Vand, nr, nr, n-1, 'N', 'C')

	for irow in range(nr):
		for icol in range(nr): #Loop on the columns of the Vandermonde matrix
			P[irow*nr + icol]  = creal(aux3C[irow*nr + icol])*creal(aux4C[irow*nr + icol])
			P[irow*nr + icol] += -creal(aux3C[irow*nr + icol])*cimag(aux4C[irow*nr + icol])*1j 
			P[irow*nr + icol] += cimag(aux3C[irow*nr + icol])*creal(aux4C[irow*nr + icol])*1j
			P[irow*nr + icol] += cimag(aux3C[irow*nr + icol])*cimag(aux4C[irow*nr + icol])
	retval = c_cholesky(P, nr)
	if not retval == 0: raiseError('Problems computing Cholesky factorization!')

	for iaux in range(nr):
		for irow in range(nr):
			aux1C[irow] = 0 + 0*1j
			for icol in range(n-1):#casting Vr to a complex, at the same time, it is multipilied per S and Vand
				aux1C[irow] += Sr[irow]*Vr[irow*(n-1) + icol]*(creal(Vand[iaux*(n-1) + icol])+cimag(Vand[iaux*(n-1) + icol])*1j)
			aux2C[irow] = w[irow*nr + iaux]
		c_zmatmult(&q[iaux], aux1C, aux2C, 1, 1, nr, 'N', 'N')

	memcpy(Pinv, P, nr*nr*sizeof(np.complex128_t))
	cdef int ii
	cdef int jj
	for ii in range(nr):
		q[ii] = creal(q[ii]) - cimag(q[ii])*1j
		for jj in range(nr - ii):
			P[ii*nr + ii+jj]   = creal(P[(ii+jj)*nr + ii])  - cimag(P[(ii+jj)*nr + ii])*1j
			P[(ii+jj)*nr + ii] = creal(Pinv[ii*nr + ii+jj]) - cimag(Pinv[ii*nr + ii+jj])*1j

	retval = c_zinverse(Pinv, nr, 'L')
	if not retval == 0: raiseError('Problems computing the Inverse!')

	c_zmatmult(aux1C, Pinv, q, nr, 1, nr, 'N', 'N')

	retval = c_zinverse(P, nr, 'U')
	if not retval == 0: raiseError('Problems computing the Inverse!')

	c_zmatmult(auxbJov, P, aux1C, nr, 1, nr, 'N', 'N')
	cr_stop('DMD.amplitudes',0)

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
	cdef np.ndarray[np.double_t,ndim=1] muReal   = np.zeros((nr),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] muImag   = np.zeros((nr),dtype=np.double)
	cdef np.ndarray[np.complex128_t,ndim=2] Phi  = np.zeros((m,nr),order='C',dtype=np.complex128)
	cdef np.ndarray[np.complex128_t,ndim=1] bJov = np.zeros((nr,),dtype=np.complex128)

	cr_start('DMD.qsort', 0)
	c_zsort(auxbJov, auxOrd, nr)
	cr_stop('DMD.qsort', 0)
	cr_start('DMD.sort', 0)
	for ii in range(nr):
		muReal[nr-(auxOrd[ii]+1)] = auxmuReal[ii]
		muImag[nr-(auxOrd[ii]+1)] = auxmuImag[ii]
		bJov[nr-(auxOrd[ii]+1)]   = auxbJov[ii]
		for jj in range(m):
			Phi[jj,nr-(auxOrd[ii]+1)]  = auxPhi[jj*nr + ii]
	cr_stop('DMD.sort', 0)

	#Free the variables that had to be ordered
	free(auxmuReal)
	free(auxmuImag)
	free(auxbJov)
	free(auxPhi)
	free(auxOrd)

	#Ensure that all conjugate modes are in the same order
	cr_start('DMD.conjugate', 0)
	cdef bint p = 0
	cdef double iimag
	for ii in range(nr):
		if p == 1:
			p = 0
			continue
		iimag = muImag[ii]
		if iimag < 0:
			muImag[ii]   =  muImag[ii+1]
			muImag[ii+1] = -muImag[ii]
			bJov[ii]     = creal(bJov[ii])   + cimag(bJov[ii+1])*1j
			bJov[ii+1]   = creal(bJov[ii+1]) - cimag(bJov[ii])*1j
			for jj in range(m):
				Phi[jj,ii]   = creal(Phi[jj,ii])   + cimag(Phi[jj,ii+1])*1j
				Phi[jj,ii+1] = creal(Phi[jj,ii+1]) - cimag(Phi[jj,ii+1])*1j
			p = 1
			continue
		if iimag > 0:
			p = 1
			continue
	cr_stop('DMD.conjugate', 0)
	
	# Return
	return muReal, muImag, Phi, bJov


## DMD frequency damping
@cr('DMD.frequency_damping')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def frequency_damping(double[:] real, double[:] imag, double dt):
	'''
	Computation of the damping ratio and the frequency of each mode
	'''
	cdef int ii, n = real.shape[0]
	cdef double mod
	cdef double arg

	cdef np.ndarray[np.double_t,ndim=1] delta = np.zeros((n),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] omega = np.zeros((n),dtype=np.double)

	for ii in range(n):
		mod       = sqrt(real[ii]*real[ii] + imag[ii]*imag[ii])
		delta[ii] = log(mod)/dt
		arg       = atan2(imag[ii], real[ii])
		omega[ii] = arg/dt
	return delta, omega

## Flow reconstruction
@cr('DMD.reconstruction_jovanovic')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def reconstruction_jovanovic(np.complex128_t[:,:] Phi, double[:] muReal, double[:] muImag, double[:] t, np.complex128_t[:] bJov):
	'''
	Computation of the reconstructed flow from the DMD computations
	'''
	cdef int m  = Phi.shape[0], n  = t.shape[0], nr = Phi.shape[1]
	cdef np.complex128_t *Vand
	cdef np.ndarray[np.complex128_t,ndim=2] Zdmd = np.zeros((m,n),order='C',dtype=np.complex128)

	Vand = <np.complex128_t*>malloc(nr*n*sizeof(np.complex128_t))

	c_vandermonde_time(Vand, &muReal[0], &muImag[0], nr, n, &t[0])
	c_zvecmat(&bJov[0], Vand, nr, n)
	c_zmatmult(&Zdmd[0,0], &Phi[0,0], Vand, m, n, nr, 'N', 'N')
	
	free(Vand)

	return Zdmd.real
