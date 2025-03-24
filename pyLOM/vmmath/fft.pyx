#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - FFT.
#
# Last rev: 27/10/2021

cimport cython
cimport numpy as np

import numpy as np

from libc.string cimport memcpy
from .cfuncs     cimport real, USE_FFTW3, c_sfft, c_dfft, c_snfft, c_dnfft, c_dhammwin
from ..utils.cr   import cr

@cr('math.hammwin')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def hammwin(int N):
	r'''
	Hamming windowing

	Args:
		N (int): Number of steps.

	Returns:
		numpy.ndarray: Hamming windowing.
	'''
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((N,), dtype=np.double)
	c_dhammwin(&out[0],N)
	return out

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _sfft(float[:] t, float[:] y, int equispaced):
	'''
	Compute the fft of a signal y that is sampled at a
	constant timestep. Return the frequency and PSD
	'''
	cdef int n = y.shape[0]
	cdef float ts = t[1] - t[0], k_left
	cdef np.ndarray[np.float32_t,ndim=1]     x
	cdef np.ndarray[np.complex64_t,ndim=1] yf
	cdef np.ndarray[np.float32_t,ndim=1] f   = np.zeros((n,), dtype=np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] PSD = np.zeros((n,), dtype=np.float32)
	memcpy(&f[0],&y[0],n*sizeof(float))
	if equispaced:
		c_sfft(&PSD[0],&f[0],ts,n)
	else:
		if USE_FFTW3:
			c_snfft(&PSD[0],&f[0],&t[0],n)
		else:
			import nfft
			# Compute sampling frequency
			k_left = (t.shape[0]-1.)/2.
			f      = (np.arange(t.shape[0],dtype=np.float32)-k_left)/t[n-1]
			# Compute power spectra using fft
			x   = -0.5 + np.arange(t.shape[0],dtype=np.float32)/t.shape[0]
			yf  = nfft.nfft_adjoint(x,y,len(t))
			PSD = np.real(yf*np.conj(yf))/y.shape[0]
	return f, PSD

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dfft(double[:] t, double[:] y, int equispaced):
	'''
	Compute the fft of a signal y that is sampled at a
	constant timestep. Return the frequency and PSD
	'''
	cdef int n = y.shape[0]
	cdef double ts = t[1] - t[0], k_left
	cdef np.ndarray[np.double_t,ndim=1]     x
	cdef np.ndarray[np.complex128_t,ndim=1] yf
	cdef np.ndarray[np.double_t,ndim=1] f   = np.zeros((n,), dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] PSD = np.zeros((n,), dtype=np.double)
	memcpy(&f[0],&y[0],n*sizeof(double))
	if equispaced:
		c_dfft(&PSD[0],&f[0],ts,n)
	else:
		if USE_FFTW3:
			c_dnfft(&PSD[0],&f[0],&t[0],n)
		else:
			import nfft
			# Compute sampling frequency
			k_left = (t.shape[0]-1.)/2.
			f      = (np.arange(t.shape[0],dtype=np.double)-k_left)/t[n-1]
			# Compute power spectra using fft
			x   = -0.5 + np.arange(t.shape[0],dtype=np.double)/t.shape[0]
			yf  = nfft.nfft_adjoint(x,y,len(t))
			PSD = np.real(yf*np.conj(yf))/y.shape[0]
	return f, PSD

@cr('math.fft')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def fft(real[:] t, real[:] y, int equispaced=True):
	r'''
	Compute the PSD of a signal y. For non equispaced time samples
	the nfft package is required.

	Args:
		t (numpy.ndarray): time vector.
		y (numpy.ndarray): signal vector.
		equispaced (bool): whether the samples in the time vector are equispaced or not.

	Returns:
		numpy.ndarray: frequency.
		numpy.ndarray: power density spectra.
	'''
	if real is double:
		return _dfft(t,y,equispaced)
	else:
		return _sfft(t,y,equispaced)