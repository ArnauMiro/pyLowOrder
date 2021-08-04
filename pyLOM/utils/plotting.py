#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Plotting - Plot routines.
#
# Last rev: 03/08/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt


def show_plots():
	'''
	Wrapper to matplotlib.pyplot.show()
	'''
	plt.show()

def close_plots():
	'''
	Wrapper to matplotlib.pyplot.close()
	'''
	plt.close()

def plotResidual(S,fig=None,ax=None):
	'''
	Given the S matrix as a vector, plot
	the accumulated residual.
	'''
	# Build figure and axes
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	# Compute accumulated residual
	accumulative_S  = np.array([np.linalg.norm(S[ii:],2) for ii in range(S.shape[0])],np.double)
	accumulative_S /= accumulative_S[0]
	# Plot accumulated residual
	ax.semilogy(np.arange(1,S.shape[0]+1),accumulative_S,'bo')
	# Set labels
	ax.set_ylabel(r'$\varepsilon_1$')
	ax.set_xlabel(r'Truncation size')
	ax.set_title(r'Tolerance')
	#ax.set_ylim((0, 1))
	# Return
	return fig, ax


def plotMode(U,y,PSD,t,mesh,fig=None,ax=None):
	'''
	Given U, VT and a mode, plot their
	representation in a figure.
	'''
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.subplots(3,1,gridspec_kw = {'hspace':0.5})
	dt = t[1] - t[0]
	# Plot the representation of mode U
	# Plot the temporal evolution of the mode
	ax[1].plot(t,y,'b')
	ax[1].set_title('Temporal mode')
	# Plot frequency representation of the mode
	freq = 1./dt/y.shape[0]*np.arange(y.shape[0])
	L    = int(np.floor(y.shape[0]/2))
	ax[2].plot(freq[:L],PSD[:L])
	ax[2].set_title('Power Spectrum')
	ax[2].set_xlabel('St')
	return fig, ax


def plotSnapshot(X,mesh,fig=None,ax=None):
	'''
	'''
	pass