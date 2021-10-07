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


def plotFieldStruct2D(ax,nx,ny,xyz,field,cmap):
	'''
	Plot a 2D field on an structured mesh
	'''
	if cmap is None: cmap = plt.get_cmap('coolwarm',256)
	X = xyz[:,0].reshape((nx,ny),order='c').T
	Y = xyz[:,1].reshape((nx,ny),order='c').T
	Z = field.reshape((nx,ny),order='c').T
	return ax.contourf(X,Y,Z,cmap=cmap)


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

def plotMode(U,xyz,V,t,PSD,freq,mesh,fig=None,ax=None,cmap=None):
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
	cf = None
	if mesh['type'] == 'struct2D': 
		cf = plotFieldStruct2D(ax[0],mesh['nx'],mesh['ny'],xyz,U,cmap)
	# Plot the temporal evolution of the mode
	ax[1].plot(t,V,'b')
	ax[1].set_title('Temporal mode')
	# Plot frequency representation of the mode
	L = int(np.floor(V.shape[0]/2))
	ax[2].plot(freq[:L],PSD[:L])
	ax[2].set_title('Power Spectrum')
	ax[2].set_xlabel('St')
	return fig, ax, cf


def plotSnapshot(X,xyz,mesh,fig=None,ax=None,cmap=None):
	'''
	Given X and the mesh plot a time instant
	'''
	# Build figure and axes
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	# Plot
	cf = plotFieldStruct2D(ax,mesh['nx'],mesh['ny'],xyz,X,cmap)
	return fig, ax, cf