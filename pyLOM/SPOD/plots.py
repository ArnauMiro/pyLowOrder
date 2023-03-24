#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# DMD plotting utilities.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from .utils        import extract_modes
from ..vmmath      import fft
from ..utils.plots import plotResidual, plotFieldStruct2D, plotSnapshot, plotLayout


def plotMode(L, P, freqs, dset, ivar, pointData=True, modes=np.array([1],np.int32),**kwargs):
	'''
	Plot the real and imaginary parts of a mode
	'''
	# Extract the modes to be plotted
	npoints = dset.mesh.size(pointData)
	P_modes = extract_modes(L,P,ivar,npoints,modes=modes)
	# Add to the dataset
	dset.add_variable('P_MODES',pointData,len(modes),P_modes)
	# Loop over the modes
	screenshot = kwargs.pop('screenshot',None)
	off_screen = kwargs.pop('off_screen',False)
	for imode, mode in enumerate(modes):
		if screenshot is not None: kwargs['screenshot'] = screenshot % imode
		plotLayout(dset,1,1,mode-1,vars=['P_MODES'],title='Mode %d St = %.3f' % (mode-1, np.abs(freqs[mode-1])),off_screen=off_screen,**kwargs)
	# Remove from dataset
	dset.delete('P_MODES')

def plotSpectra(f, L, fig=None, ax=None):
	# Get or recover axis and figure
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	# Plot
	for ii in range(L.shape[1]):
		ax.loglog(np.sort(f), L[np.argsort(f),ii], 'o-')
	ax.set_xlabel('St')
	ax.set_ylabel(r'$\lambda_i$')
	return fig, ax