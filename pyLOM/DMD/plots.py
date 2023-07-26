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


def plotMode(Phi, omega, dset, ivar, pointData=True, modes=np.array([1],np.int32),**kwargs):
	'''
	Plot the real and imaginary parts of a mode
	'''
	# Extract the modes to be plotted
	npoints = dset.mesh.size(pointData)
	Phi_real = extract_modes(Phi,ivar,npoints,real=True,modes=modes)
	Phi_imag = extract_modes(Phi,ivar,npoints,real=False,modes=modes)
	# Add to the dataset
	dset.add_variable('PHI_REAL',pointData,len(modes),Phi_real)
	dset.add_variable('PHI_IMAG',pointData,len(modes),Phi_imag)	
	# Loop over the modes
	screenshot = kwargs.pop('screenshot',None)
	off_screen = kwargs.pop('off_screen',False)
	for imode, mode in enumerate(modes):
		if screenshot is not None: kwargs['screenshot'] = screenshot % imode
		plotLayout(dset,2,1,mode-1,vars=['PHI_REAL','PHI_IMAG'],title='Mode %d St = %.3f' % (mode-1, np.abs(omega[mode-1])/(2*np.pi)),off_screen=off_screen,**kwargs)
	# Remove from dataset
	dset.delete('PHI_REAL')
	dset.delete('PHI_IMAG')


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
