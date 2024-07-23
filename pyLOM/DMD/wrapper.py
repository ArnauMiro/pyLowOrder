#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for DMD.
#
# Last rev: 30/09/2021
from __future__ import print_function

import numpy as np
from ..vmmath       import vector_norm, vecmat, matmul, temporal_mean, subtract_mean, tsqr_svd, transpose, eigen, cholesky, diag, polar, vandermonde, conj, inv, flip, matmulp, vandermondeTime, variable_projection_optimizer
from ..POD          import truncate
from ..utils.cr     import cr, cr_start, cr_stop
from ..utils.errors import raiseError
from ..utils.parall import mpi_gather, mpi_reduce, pprint, MPI_SIZE

@cr('DMD.order_modes')
def _order_modes(muReal, muImag, Phi, b):
	'''
    Order the modes according to its amplitude, forcing that in case of a conjugate eigenvalue, the positive part always is the first one
	'''
	muReal = muReal[flip(np.abs(b).argsort())]
	muImag = muImag[flip(np.abs(b).argsort())]
	Phi    = transpose(transpose(Phi)[flip(np.abs(b).argsort())])
	b      = b[flip(np.abs(b).argsort())]
	p = False
	for ii in range(muImag.shape[0]):
		if p == True:
			p = False
			continue
		iimag = muImag[ii]
		if iimag < 0:
			muImag[ii]       =  muImag[ii+1]
			muImag[ii+1]     = -muImag[ii]
			b[ii]            =  b.real[ii]   + b.imag[ii+1]
			b[ii+1]          =  b.real[ii+1] - b.imag[ii]
			Phi.imag[:,ii]   =  Phi.imag[:,ii+1]
			Phi.imag[:,ii+1] = -Phi.imag[:,ii+1]
			p = True
			continue
		if iimag > 0:
			p = True
			continue
	return muReal, muImag, Phi, b

@cr('DMD.optimized_modes')
def _optimized_modes(U, S, VT, Atilde, t):
	
	#Compute target to optimize
	H = np.matmul(np.diag(S), VT).T

	#Eigendecomposition of Atilde for initailization of the variable projection optimization algorithm
	iniReal, iniImag, iniW=  eigen(Atilde)
	#Optimization of eigenvalues
	B, alpha = variable_projection_optimizer(H, iniReal, iniImag, t[:H.shape[0]])
	#Compute modes and amplitudes
	muReal = alpha.real
	muImag = alpha.imag
	w      = B.T
	b      = np.sqrt(np.sum(np.abs(w) ** 2, axis=0))
	w      = w.dot(np.diag(1/b))
	Phi    = U.dot(w)

	return muReal, muImag, Phi, b

@cr('DMD.run')
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
	#Remove temporal mean or not, depending on the user choice
	if remove_mean:
		cr_start('DMD.temporal_mean',0)
		#Compute temporal mean
		X_mean = temporal_mean(X)
		#Subtract temporal mean
		Y = subtract_mean(X, X_mean)
		cr_stop('DMD.temporal_mean',0)
	else:
		Y = X.copy()

	#Compute SVD
	cr_start('DMD.SVD',0)
	U, S, VT = tsqr_svd(Y[:, :-1])
	cr_stop('DMD.SVD',0)

	# Truncate according to residual
	cr_start('DMD.truncate', 0)
	U, S, VT = truncate(U, S, VT, r)
	cr_stop('DMD.truncate', 0)

	#Project A (Jacobian of the snapshots) into POD basis
	cr_start('DMD.linear_mapping',0)
	aux1   = matmulp(transpose(U), Y[:, 1:])
	aux2   = transpose(vecmat(1./S, VT))
	Atilde = matmul(aux1, aux2)
	cr_stop('DMD.linear_mapping',0)

	cr_start('DMD.modes',0)
	muReal, muImag, w = eigen(Atilde)

	#Mode computation
	Phi = matmul(matmul(matmul(Y[:, 1:], transpose(VT)), diag(1/S)), w)/(muReal + muImag*1J)
	cr_stop('DMD.modes',0)

	#Amplitudes according to: Jovanovic et. al. 2014 DOI: 10.1063
	cr_start('DMD.amplitudes',0)
	Vand = vandermonde(muReal, muImag, muReal.shape[0], Y.shape[1]-1)
	P    = matmul(transpose(conj(w)), w)*conj(matmul(Vand, transpose(conj(Vand))))
	Pl   = cholesky(P)
	G    = matmul(diag(S), VT)
	q    = conj(diag(matmul(matmul(Vand, transpose(conj(G))), w)))
	bJov = matmul(inv(transpose(conj(Pl))), matmul(inv(Pl), q)) #Amplitudes according to Jovanovic 2014
	cr_stop('DMD.amplitudes',0)

	#Order modes and eigenvalues according to its amplitude
	if muReal.shape[0] == 1:
		return muReal, muImag, Phi, bJov
	else:
		cr_start('DMD.order',0)
		muReal, muImag, Phi, bJov = _order_modes(muReal, muImag, Phi, bJov)
		cr_stop('DMD.order',0)
		return muReal, muImag, Phi, bJov

@cr('DMD.run_optimized')
def run_optimized(X, t, r, constraints=None, remove_mean=True):
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
	#Remove temporal mean or not, depending on the user choice
	if remove_mean:
		cr_start('DMD.temporal_mean',0)
		#Compute temporal mean
		X_mean = temporal_mean(X)
		#Subtract temporal mean
		Y = subtract_mean(X, X_mean)
		cr_stop('DMD.temporal_mean',0)
	else:
		Y = X.copy()

	##Initialize the eigenvalues using the exact DMD
	#Compute SVD
	cr_start('DMD.SVD',0)
	U, S, VT = tsqr_svd(Y[:, :-1])
	cr_stop('DMD.SVD',0)
	# Truncate according to residual
	cr_start('DMD.truncate', 0)
	U, S, VT = truncate(U, S, VT, r)
	cr_stop('DMD.truncate', 0)

	#Project A (Jacobian of the snapshots) into POD basis
	cr_start('DMD.linear_mapping',0)
	aux1   = matmulp(transpose(U), Y[:, 1:])
	aux2   = transpose(vecmat(1./S, VT))
	Atilde = matmul(aux1, aux2)
	cr_stop('DMD.linear_mapping',0)

	##Compute the optimized eigenvalues and modes and its amplitudes
	muReal, muImag, Phi, b = _optimized_modes(U, S, VT, Atilde, t)
	#Order modes and eigenvalues according to its amplitude
	if muReal.shape[0] == 1:
		return muReal, muImag, Phi, b
	else:
		muReal, muImag, Phi, b = _order_modes(muReal, muImag, Phi, b)
		return muReal, muImag, Phi, b

@cr('DMD.frequency_damping')
def frequency_damping(real, imag, dt):
	'''
	Computation of the damping ratio and the frequency of each mode
	'''
	mod, arg = polar(real, imag) #Create vmmath/complex.c?
	#Computation of the damping ratio of the mode
	delta = np.log(mod)/dt
	#Computation of the frequency of the mode
	omega = arg/dt
	return delta, omega

@cr('DMD.mode_computation')
def mode_computation(X, V, S, W):
	'''
	Computation of DMD Modes
	'''
	return  matmul(matmul(matmul(X, transpose(V)), diag(1/S)), np.abs(W))

@cr('DMD.reconstruction_jovanovic')
def reconstruction_jovanovic(Phi, real, imag, t, bJov):
	'''
    Reconstruction of the DMD modes according to the Jovanovic method
	'''
	Vand = vandermondeTime(real, imag, real.shape[0], t)
	return matmul(Phi, matmul(diag(bJov), Vand)).real
