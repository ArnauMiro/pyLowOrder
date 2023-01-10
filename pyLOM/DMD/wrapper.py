#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for DMD.
#
# Last rev: 30/09/2021
from __future__ import print_function

import numpy as np
from ..vmmath       import vector_norm, vecmat, matmul, temporal_mean, subtract_mean, tsqr_svd, transpose, eigen, cholesky, diag, polar, vandermonde, conj, inv, flip, matmul_paral, vandermondeTime
from ..POD          import truncate
from ..utils.cr     import cr_start, cr_stop
from ..utils.errors import raiseError
from ..utils.parall import mpi_gather, mpi_reduce, pprint

def run(X, r, remove_mean = True):
	'''
	DMD analysis of snapshot matrix X
	Inputs:
		- X[ndims*nmesh,n_temp_snapshots]: data matrix
		- remove_mean:                     whether or not to remove the mean flow

	Returns:
		- Phi:      DMD Modes
		- muReal:   Real part of the eigenvalues
		- muImag:   Imaginary part of the eigenvalues
		- b:        Amplitude of the DMD modes
		- X_DMD:    Reconstructed flow
	'''
	cr_start('DMD.run', 0)
	#Remove temporal mean or not, depending on the user choice
	if remove_mean:
		#Compute temporal mean
		X_mean = temporal_mean(X)
		#Subtract temporal mean
		Y = subtract_mean(X, X_mean)
	else:
		Y = X.copy()

	#Compute SVD
	U, S, VT = tsqr_svd(Y[:, :-1])
	# Truncate according to residual
	U, S, VT = truncate(U, S, VT, r)

	#Project A (Jacobian of the snapshots) into POD basis
	aux1   = matmul_paral(transpose(U), Y[:, 1:])
	aux2   = transpose(vecmat(1./S, VT))
	Atilde = matmul(aux1, aux2)

	#Eigendecomposition of Atilde: Eigenvectors given as complex matrix
	muReal, muImag, w = eigen(Atilde)

	#Mode computation
	Phi =  matmul(matmul(matmul(Y[:, 1:], transpose(VT)), diag(1/S)), w)/(muReal + muImag*1J)

	#Amplitudes according to: Jovanovic et. al. 2014 DOI: 10.1063
	Vand = vandermonde(muReal, muImag, muReal.shape[0], Y.shape[1]-1)
	P    = matmul(transpose(conj(w)), w)*conj(matmul(Vand, transpose(conj(Vand))))
	Pl   = cholesky(P)
	G    = matmul(diag(S), VT)
	q    = conj(diag(matmul(matmul(Vand, transpose(conj(G))), w)))
	bJov = matmul(inv(transpose(conj(Pl))), matmul(inv(Pl), q)) #Amplitudes according to Jovanovic 2014

	#Order modes and eigenvalues according to its amplitude 
	muReal, muImag, Phi, bJov = order_modes(muReal, muImag, Phi, bJov)

	cr_stop('DMD.run', 0)

	return muReal, muImag, Phi, bJov

def frequency_damping(real, imag, dt):
	'''
	Computation of the damping ratio and the frequency of each mode
	'''
	cr_start('DMD.frequency_damping', 0)
	mod, arg = polar(real, imag) #Create vmmath/complex.c?
	#Computation of the damping ratio of the mode
	delta = np.log(mod)/dt
	#Computation of the frequency of the mode
	omega = arg/dt
	cr_stop('DMD.frequency_damping', 0)
	return delta, omega

def mode_computation(X, V, S, W):
	'''
	Computation of DMD Modes
	'''
	cr_start('DMD.mode_computation', 0)
	Phi =  matmul(matmul(matmul(X, transpose(V)), diag(1/S)), np.abs(W))
	cr_stop('DMD.mode_computation', 0)
	return Phi

def amplitude_jovanovic(real, imag, X1, wComplex, S, V):
	'''
    Computation of the amplitude of the DMD modes according to Jovanovic et. al. 2014 DOI: 10.1063
    '''
	cr_start('DMD.amplitude_jovanovic', 0)
	Vand = vandermonde(real, imag, real.shape[0], X1.shape[1])
	P    = matmul(transpose(conj(wComplex)), wComplex)*conj(matmul(Vand, transpose(conj(Vand))))
	Pl   = cholesky(P)
	G    = matmul(diag(S), V)
	q    = conj(diag(matmul(matmul(Vand, transpose(conj(G))), wComplex)))
	bJov = matmul(inv(transpose(conj(Pl))), matmul(inv(Pl), q)) #Amplitudes according to Jovanovic 2014
	cr_stop('DMD.amplitude_jovanovic', 0)
	return bJov

def reconstruction_jovanovic(Phi, real, imag, t, bJov):
	'''
    Reconstruction of the DMD modes according to the Jovanovic method
	'''
	cr_start('DMD.reconstruction_jovanovic', 0)
	Vand = vandermondeTime(real, imag, real.shape[0], t)
	Xdmd = matmul(Phi, matmul(diag(bJov), Vand))
	cr_stop('DMD.reconstruction_jovanovic', 0)
	return Xdmd.real

def order_modes(muReal, muImag, Phi, bJov):
	'''
    Order the modes according to its amplitude, forcing that in case of a conjugate eigenvalue, the positive part always is the first one
	'''
	cr_start('DMD.order_modes', 0)
	muReal = muReal[flip(np.abs(bJov).argsort())]
	muImag = muImag[flip(np.abs(bJov).argsort())]
	Phi    = transpose(transpose(Phi)[flip(np.abs(bJov).argsort())])
	bJov   = bJov[flip(np.abs(bJov).argsort())]
	p = False
	for ii in range(muImag.shape[0]):
		if p == True:
			p = False
			continue
		iimag = muImag[ii]
		if iimag < 0:
			muImag[ii]        =  muImag[ii+1]
			muImag[ii+1]      = -muImag[ii]
			bJov.imag[ii]     =  bJov.imag[ii+1]
			bJov.imag[ii+1]   = -bJov.imag[ii]
			Phi.imag[:,ii]    =  Phi.imag[:,ii+1]
			Phi.imag[:,ii+1]  = -Phi.imag[:,ii+1]
			p = True
			continue
		if iimag > 0:
			p = True
			continue
	cr_stop('DMD.order_modes', 0)
	return muReal, muImag, Phi, abJov