#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# POD plotting utilities.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from ..vmmath      import fft
from ..utils       import gpu_to_cpu


def plotMode(V:np.ndarray,t:np.ndarray,modes:np.ndarray=np.array([1],np.int32),fftfun:object=fft,scale_freq:np.double=1.,fig:plt.figure=[],ax:plt.axes=[]):
	r'''
	Plot the temporal coefficient and its frequency spectrum of a set of modes

	Args:
		V (np.ndarray): array containing the temporal coefficients from the POD modes
		t (np.ndarray): array containing the time values
		modes (np.ndarray): array with the modes to plot
		fftfun (object, optional): function to use to compute the frequency spectra (default: ``fft``)
		scale_freq (double, optional): value used to non-dimensionalize the frequencies (default: ``1``)
		fig (plt.figure, optional): figure object in which the plot will be done (default: ``[]``)
		axs (plt.axes, optional): axes object in which the plot will be done (default: ``[]``)

	Returns:
		[plt.figure, plt.axes]: figure and axes objects of the plot
	'''
	V = gpu_to_cpu(V)
	for imode, mode in enumerate(modes):
		if len(fig) < imode + 1:
			fig.append( plt.figure(figsize=(8,6),dpi=100) )
		if len(ax) < imode + 1:
			ax.append( fig[imode].subplots(2,1,gridspec_kw = {'hspace':0.5}) )
		fig[imode].suptitle('Mode %d'%(mode-1))
	   	# Plot the temporal evolution of the mode
		ax[imode][0].plot(t,V[mode-1,:],'b')
		ax[imode][0].set_title('Temporal mode')
		# Plot frequency representation of the mode
		if V.shape[1] % 2 == 0:
			freq,psd = fftfun(t,V[mode-1,:],equispaced=False)
		else:
			freq,psd = fftfun(t[:-1],V[mode-1,:-1],equispaced=False)
		freq *= scale_freq
		#L = int(np.floor(freq.shape[0]/2))
		ax[imode][1].plot(freq,psd, 'b')
		ax[imode][1].set_title('Power Spectrum')
		ax[imode][1].set_xlabel('St')
		ax[imode][1].set_xlim([0,1])
	return fig, ax

def plotEnergy(S,fig=None,ax=None):
	r'''
	Plot the cummulative energy of a set of modes

	Args:
		S (np.ndarray): array containing the singular values from the POD modes
		fig (plt.figure, optional): figure object in which the plot will be done (default: ``[]``)
		axs (plt.axes, optional): axes object in which the plot will be done (default: ``[]``)

	Returns:
		[plt.figure, plt.axes]: figure and axes objects of the plot
	'''
	S = gpu_to_cpu(S)
	# Build figure and axes
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	# Plot accumulated residual
	ax.plot(np.arange(1,S.shape[0]+1),np.cumsum(S**2)/np.sum(S**2),'bo')
	# Set labels
	ax.set_ylabel(r'Energy')
	ax.set_xlabel(r'Modes')
	# Return
	return fig, ax