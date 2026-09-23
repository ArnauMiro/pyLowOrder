#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Resolvent analysis plotting utilities.
#
# Last rev: 08/04/2026
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from ..vmmath      import fft
from ..utils       import gpu_to_cpu


def plotEnergy(S,fig=None,ax=None):
	r'''
	Plot the cummulative energy of a set of modes

	Args:
		S (np.ndarray): array containing the singular values from the RES modes
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

def plotEvW(S, omega, modes, fig=None,ax=None):
	r'''
	Plot the energy gain for different frequencies

	Args:
		S (np.ndarray): array containing the energy gains from the RES modes (columns) at diferent frequencies (rows)
		omega (np.ndarray): array containing the frequencies from the respective energy gains
		modes (np.ndarray): modes to plot
		fig (plt.figure, optional): figure object in which the plot will be done (default: ``[]``)
		axs (plt.axes, optional): axes object in which the plot will be done (default: ``[]``)

	Returns:
		[plt.figure, plt.axes]: figure and axes objects of the plot
	'''
	S = gpu_to_cpu(S)
	omega = gpu_to_cpu(omega)
	# Build figure and axes
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	# Plot energy gains
	for ii in modes:
		ax.plot(omega, S[:,ii], 'o-', label=f'mode {ii}')
	# Set labels
	ax.set_ylabel(r'Energy')
	ax.set_xlabel(r'Frequencies')
	ax.legend()
	# Return
	return fig, ax