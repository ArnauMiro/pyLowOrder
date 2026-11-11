#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for DMD.
#
# Last rev: 30/09/2021
from __future__ import print_function

import numpy as np

from ..utils.gpu import cp
from ..vmmath    import matmul, temporal_mean, subtract_mean, transpose, eigen, cholesky, diag, polar, vandermonde, conj, inv, vandermondeTime, linear_operator, concatenate, separate
from ..utils     import cr_nvtx as cr, cr_start, cr_stop


def _order_modes(muReal, muImag, Phi, bJov):
	'''
    Order the modes according to its amplitude, forcing that in case of a conjugate eigenvalue, the positive part always is the first one
	'''
	cnp    = cp if type(muReal) is cp.ndarray else np
	# NOTE: vmmath's flip() is not implemented; reversed slicing is equivalent
	idx    = cnp.abs(bJov).argsort()[::-1]
	muReal = muReal[idx]
	muImag = muImag[idx]
	Phi    = transpose(transpose(Phi)[idx])
	bJov   = bJov[idx]
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
	return muReal, muImag, Phi, bJov

@cr('DMD.run')
def run(X, r, remove_mean = True):
	'''
	DMD analysis of snapshot matrix X or list of snapshot matrices X
	Inputs:
		- X[ndims*nmesh,n_temp_snapshots]: data matrix or list of data matrices
		                                   with stitched interfaces, i.e.
		                                   X[i+1][:,0] == X[i][:,-1], sharing
		                                   the same spatial dimension
		- remove_mean:                     whether or not to remove the mean flow
		- r:                               maximum truncation residual

	Returns:
		- Phi:      DMD Modes
		- muReal:   Real part of the eigenvalues
		- muImag:   Imaginary part of the eigenvalues
		- b:        Amplitude of the DMD modes
		- X_DMD:    Reconstructed flow
	'''
	# Prepare matrices and calculate the linear operator
	if (type(X) is list):
		Y, Z = concatenate(X, remove_mean=remove_mean)
	else:
		Y, Z = separate(X, remove_mean=remove_mean)
	U, S, VT, Atilde = linear_operator(Y, Z, r)

	del U

	# Eigendecomposition of Atilde: Eigenvectors given as complex matrix
	# NOTE: there is no implementation of eig in cupy yet
	cr_start('DMD.modes',0)
	muReal, muImag, w = eigen(Atilde)
	muReal = cp.asarray(muReal) if type(Atilde) is cp.ndarray else muReal
	muImag = cp.asarray(muImag) if type(Atilde) is cp.ndarray else muImag
	w      = cp.asarray(w)      if type(Atilde) is cp.ndarray else w

	# Mode computation
	# NOTE: the real intermediate is promoted to complex since vmmath's fused
	# matmul requires both operands to share the same dtype
	Vr   = matmul(matmul(Z, transpose(VT)), diag(1/S))
	Phi  = matmul(Vr.astype(np.result_type(Vr.dtype, w.dtype)), w)/(muReal + muImag*1J)
	cr_stop('DMD.modes',0)

	# Amplitudes according to: Jovanovic et. al. 2014 DOI: 10.1063
	cr_start('DMD.amplitudes',0)
	Vand = vandermonde(muReal, muImag, muReal.shape[0], Y.shape[1])
	P    = matmul(transpose(conj(w)), w)*conj(matmul(Vand, transpose(conj(Vand))))
	Pl   = cholesky(P)
	# NOTE: G is real but is promoted to complex since vmmath's fused matmul
	# requires both operands to share the same dtype (the compiled core folds
	# this product into its complex loops)
	G    = matmul(diag(S), VT).astype(w.dtype)
	# NOTE: vmmath's conj and matmul only accept matrices, so the vector q is
	# conjugated with numpy and passed to matmul as a column matrix (the
	# compiled core does the same, treating q as nr x 1)
	q    = np.conj(diag(matmul(matmul(Vand, transpose(conj(G))), w)))
	bJov = matmul(inv(transpose(conj(Pl))), matmul(inv(Pl), q.reshape(-1,1))).ravel() #Amplitudes according to Jovanovic 2014
	cr_stop('DMD.amplitudes',0)

	# Order modes and eigenvalues according to its amplitude
	cr_start('DMD.order',0)
	muReal, muImag, Phi, bJov = _order_modes(muReal, muImag, Phi, bJov)
	cr_stop('DMD.order',0)

	return muReal, muImag, Phi, bJov

@cr('DMD.frequency_damping')
def frequency_damping(real, imag, dt):
	'''
	Computation of the damping ratio and the frequency of each mode
	'''
	p = cp if type(real) is cp.ndarray else np
	mod, arg = polar(real, imag) #Create vmmath/complex.c?
	# Computation of the damping ratio of the mode
	delta = p.log(mod)/dt
	# Computation of the frequency of the mode
	omega = arg/dt
	return delta, omega

@cr('DMD.mode_computation')
def mode_computation(X, V, S, W):
	'''
	Computation of DMD Modes
	'''
	p = cp if type(X) is cp.ndarray else np
	return  matmul(matmul(matmul(X, transpose(V)), diag(1/S)), p.abs(W))

@cr('DMD.reconstruction_jovanovic')
def reconstruction_jovanovic(Phi, real, imag, t, bJov):
	'''
    Reconstruction of the DMD modes according to the Jovanovic method
	'''
	Vand = vandermondeTime(real, imag, real.shape[0], t)
	return matmul(Phi, matmul(diag(bJov), Vand)).real
