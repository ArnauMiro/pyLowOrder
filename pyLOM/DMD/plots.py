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

from ..vmmath      import fft
from ..utils.plots import plotResidual, plotFieldStruct2D, plotSnapshot, animateFlow
from ..utils.mesh  import mesh_compute_cellcenter


def plotMode(Phi, omega, xyz,mesh,info,dim=0,modes=np.array([1],np.int32),scale_freq=1.,fig=[],ax=[],cmap=None):
	'''
	Given U, VT and a mode, plot their
	representation in a figure.
	'''
	cf = []
	for imode, mode in enumerate(modes):
		if len(fig) < imode + 1:
			fig.append( plt.figure(figsize=(8,6),dpi=100) )
		if len(ax) < imode + 1:
			ax.append( fig[imode].subplots(2,1,gridspec_kw = {'hspace':0.5}) )
		fig[imode].suptitle('Mode %d St = %.3f' % (mode, np.abs(omega[mode-1])/(2*np.pi)))
		if mesh['type'] == 'struct2D':
			c = None
			if info['point']:
				c = plotFieldStruct2D(ax[imode][0],mesh['nx'],mesh['ny'],info['ndim'],xyz,Phi[:,mode-1].real,dim-1,cmap)
				c = plotFieldStruct2D(ax[imode][1],mesh['nx'],mesh['ny'],info['ndim'],xyz,Phi[:,mode-1].imag,dim-1,cmap)
			else:
				xyzc = mesh_compute_cellcenter(xyz,mesh)
				c = plotFieldStruct2D(ax[imode][0],mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,Phi[:,mode-1].real,dim-1,cmap)
				c = plotFieldStruct2D(ax[imode][1],mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,Phi[:,mode-1].imag,dim-1,cmap)
			cf.append(c)
		ax[imode][0].set_title('Real mode')
		ax[imode][1].set_title('Imaginary mode')
	return fig, ax, cf

def ritzSpectrum(real, imag, fig = None, ax = None, cmap = None):
	'''
	Given the real and imaginary part of the eigenvalues, plot the Ritz Spectrum together with the unit circle
	'''
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	if cmap is None:
		cmap = 'r'
	theta = np.array(range(101))*2*np.pi/100
	ax.plot(np.cos(theta), np.sin(theta), c = 'k')
	ax.scatter(real, imag, c = cmap)
	ax.axis('equal')
	ax.set(xlabel = '$\mu_{Re}$', ylabel = '$\mu_{Imag}$', title = 'Ritz spectrum')
	return fig, ax

def amplitudeFrequency(omega, amplitude, fig = None, ax = None, cmap = None, mark = None, norm = False):
	'''
	Given the frequency and the amplitude of the DMD modes, plot the amplitude against the Strouhal number
	'''
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	if cmap is None:
		cmap = 'r'
	if mark is None:
		mark = 'X'
	if norm is True:
		amplitude = np.abs(amplitude)/np.max(np.abs(amplitude))
	else:
		amplitude = np.abs(amplitude)
	ax.scatter(omega/(2*np.pi), amplitude, c = cmap, marker = mark)
	ax.set(xlabel = 'St', ylabel = 'Amplitude', title = 'Amplitude vs Frequency of the DMD Modes')
	ax.set_yscale('log')
	return fig, ax

def dampingFrequency(omega, delta, fig = None, ax = None, cmap = None, mark = None):
	'''
	Given the frequency and the damping ratio of the DMD modes, plot the amplitude against the Strouhal number
	'''
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	if cmap is None:
		cmap = 'r'
	if mark is None:
		mark = 'X'
	ax.scatter(omega/(2*np.pi), np.abs(delta), c = cmap, marker = mark)
	ax.set(xlabel = 'St', ylabel = 'Damping Ratio', title = 'Damping ratio vs Frequency of the DMD Modes')
	ax.set_yscale('log')
	return fig, ax
