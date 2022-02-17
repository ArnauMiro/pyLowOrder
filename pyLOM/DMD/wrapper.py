#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for DMD.
#
# Last rev: 30/09/2021
from __future__ import print_function

import numpy as np

from ..vmmath       import vector_norm, vecmat, matmul, temporal_mean, subtract_mean, tsqr_svd, polar, transpose, diag, vandermonde
from ..utils.cr     import cr_start, cr_stop
from ..utils.errors import raiseError

def frequency_damping(real, imag, dt):
	'''
	Computation of the damping ratio and the frequency of each mode
	'''
	mod, arg = polar(real, imag)
	#Computation of the damping ratio of the mode
	delta = np.log(mod)/dt
	#Computation of the frequency of the mode
	omega = arg/dt
	return delta, omega

def mode_computation(X, V, S, W):
	'''
	Computation of DMD Modes
	'''
	cr_start('DMD.mode_computation', 0)
	Phi =  matmul(matmul(matmul(X, transpose(V)), diag(1/S)), W)
	cr_stop('DMD.mode_computation', 0)
	return Phi

def amplitude_jovanovic(real, imag, X1, wComplex, S, V): #CUIDAO amb el try-except
	'''
    Computation of the amplitude of the DMD modes according to the Jovanovic method
    '''
	cr_start('DMD.amplitude_jovanovic', 0)
	Vand = vandermonde(real, imag, real.shape[0], X1.shape[1])
	P    = matmul(transpose(conj(wComplex)), wComplex)*conj(matmul(Vand, transpose(conj(Vand))))
	Pl = cholesky(P)
	G    = matmul(diag(S), V)
	q    = conj(diag(matmul(matmul(Vand, transpose(conj(G))), wComplex)))
	bJov = matmul(inv(transpose(conj(Pl))), matmul(inv(Pl), q)) #Amplitudes according to Jovanovic 2014
	cr_stop('DMD.amplitude_jovanovic', 0)
	return bJov

def mode_computation(X, V, S, W):
	'''
	Computation of DMD Modes
	'''
	cr_start('DMD.mode_computation', 0)
	Phi =  matmul(matmul(matmul(X, transpose(V)), diag(1/S)), W)
	cr_stop('DMD.math.mode_computation', 0)
	return Phi
'''
def reconstruction_jovanovic(U, w, real, imag, X1, bJov):
    Reconstruction of the DMD modes according to the Jovanovic method
    cr_start('DMD.reconstruction_jovanovic', 0)
	Vand = vandermonde(real, imag, real.shape[0], X1.shape[1])
    Xdmd = matmul(matmul(matmul(PSI, w), diag(bJov)), Vand)
    cr_stop('DMD.reconstruction_jovanovic', 0)
    return Xdmd

def order_modes(delta, omega, Phi, amp):
    Order the modes according to its amplitude
    cr_start('DMD.order_modes', 0)
    delta  = delta[np.flip(np.abs(amp).argsort())]
	omega  = omega[np.flip(np.abs(amp).argsort())]
	Phi    = transpose(transpose(Phi)[np.flip(np.abs(amp).argsort())])
	amp   = amp[np.flip(np.abs(amp).argsort())]
    cr_stop('DMD.order_modes', 0)
	return delta, omega, Phi, amp
'''
