#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - FFT.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np, scipy, nfft

from .maths     import conj
from ..utils.cr import cr_nvtx as cr


@cr('math.hammwin')
def hammwin(N):
	'''
	Hamming windowing
	'''
	return np.transpose(0.54-0.46*np.cos(2*np.pi*np.arange(N)/(N-1)))

@cr('math.fft')
def fft(t,y,equispaced=True):
	'''
	Compute the PSD of a signal y.
	'''
	if equispaced:
		ts = t[1] - t[0] # Sampling time
		# Compute sampling frequency
		f  = 1./ts/t.shape[0]*np.arange(t.shape[0],dtype=y.dtype)
		# Compute power spectra using fft
		yf = scipy.fft.fft(y)
	else:
		# Compute sampling frequency
		k_left = (t.shape[0]-1.)/2.
		f      = (np.arange(t.shape[0],dtype=y.dtype)-k_left)/t[-1]
		# Compute power spectra using fft
		x  = -0.5 + np.arange(t.shape[0],dtype=y.dtype)/t.shape[0]
		yf = nfft.nfft_adjoint(x,y,len(t))
	ps = np.real(yf*conj(yf))/y.shape[0] # np.abs(yf)/y.shape[0]
	return f, ps