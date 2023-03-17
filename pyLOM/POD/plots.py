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
from ..utils.plots import plotResidual, plotSnapshot


def plotMode(V,t,modes=np.array([1],np.int32),fftfun=fft,scale_freq=1.,fig=[],ax=[],cmap=None):
	'''
	Given U, VT and a mode, plot their
	representation in a figure.
	'''
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