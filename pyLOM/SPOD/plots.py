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
from ..utils       import gpu_to_cpu
from ..utils.plots import plotLayout
from ..vmmath      import vecmat, vector_sum, abs
from ..utils.gpu   import cp
from ..            import Mesh, Dataset


def plotMode(L:np.ndarray, P:np.ndarray, freqs:np.ndarray, mesh:Mesh, dset:Dataset, ivar:int, pointData:bool=True, modes:np.ndarray=np.array([1],np.int32),**kwargs):
	r'''
	Plot a SPOD mode, including both the real and imaginary parts

	Args:
		L (np.ndarray): modal energy spectra
		P (np.ndarray): spatial SPOD modes
		freqs (np.ndarray): frequencies at which the modes are computed
		mesh (Mesh): mesh at which the data is represented
		dset (Dataset): pyLOM Dataset containing the case information
		ivar (int): index of the variable inside the dataset
		pointData(bool, optional): whether the data is represented on points or cell (default, ``True``)
		modes (np.ndarray, optinal): IDs of the modes to plot
	'''
	L, P = gpu_to_cpu(L), gpu_to_cpu(P)
	# Extract the modes to be plotted
	npoints = mesh.size(pointData)
	P_modes = extract_modes(L,P,ivar,npoints,modes=modes)
	# Add to the dataset
	dset.add_field('P_MODES',len(modes),P_modes)
	# Loop over the modes
	screenshot = kwargs.pop('screenshot',None)
	off_screen = kwargs.pop('off_screen',False)
	for imode, mode in enumerate(modes):
		if screenshot is not None: kwargs['screenshot'] = screenshot % imode
		plotLayout(mesh,dset,1,1,mode-1,vars=['P_MODES'],title='Mode %d St = %.3f' % (mode-1, np.abs(freqs[mode-1])),off_screen=off_screen,**kwargs)
	# Remove from dataset
	dset.delete('P_MODES')

def plotSpectra(f:np.ndarray, L:np.ndarray, fig:plt.figure=None, ax:plt.axes=None):
	r'''
	Plot the frequency-enegy spectrum

	Args:
		f (np.ndarray): frequencies at which the modes are computed
		L (np.ndarray): energy of each frequency
		fig (plt.figure, optional): figure object in which the plot will be done (default: ``[]``)
		axs (plt.axes, optional): axes object in which the plot will be done (default: ``[]``)

	Returns:
		[plt.figure, plt.axes]: figure and axes objects of the plot

	'''
	L, f = gpu_to_cpu(L), gpu_to_cpu(f)
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


def plotConvergence(PRef, PRed1, PRed2, W=None, f=None, fig:plt.figure=None, ax:plt.axes=None):
	r'''
	Plot the convergence of the SPOD modes of a block with respect to the reference ones.
	The modes of the same block must have been extracted with extract_modes from runs using
	the same nDFT, nolap and dt so that the frequencies match. 

	Args:
		PRef (np.ndarray): reference modes, (npoints, nmodes) array as returned by extract_modes with reshape=False
		PRed1 (np.ndarray): modes of the first reduced dataset, (npoints, nmodes) array as returned by extract_modes with reshape=False
		PRed2 (np.ndarray): modes of the second reduced dataset, (npoints, nmodes) array as returned by extract_modes with reshape=False
		W (np.ndarray, optional): diagonal of the inner product weighting matrix (default: ``None``)
		f (np.ndarray, optional): frequencies corresponding to the columns of the modes, used as x axis (default: ``None``, i. e. mode id)
		fig (plt.figure, optional): figure object in which the plot will be done (default: ``[]``)
		ax (plt.axes, optional): axes object in which the plot will be done (default: ``[]``)

	Returns:
		[plt.figure, plt.axes]: figure and axes objects of the plot
	'''
	p = cp if type(PRef) is cp.ndarray else np
	if W is not None:
		PRef  = vecmat(p.sqrt(W), PRef)
		PRed1 = vecmat(p.sqrt(W), PRed1)
		PRed2 = vecmat(p.sqrt(W), PRed2)

	nmodes = PRef.shape[1]
	G1 = p.zeros(nmodes) 
	G2 = p.zeros(nmodes)
	Rf = p.zeros(nmodes) 
	R1 = p.zeros(nmodes)
	R2 = p.zeros(nmodes)
	for ii in range(nmodes):
		G1[ii] = vector_sum(PRef[:,ii]  * PRed1[:,ii])
		G2[ii] = vector_sum(PRef[:,ii]  * PRed2[:,ii])
		Rf[ii] = vector_sum(PRef[:,ii]  * PRef[:,ii])
		R1[ii] = vector_sum(PRed1[:,ii] * PRed1[:,ii])
		R2[ii] = vector_sum(PRed2[:,ii] * PRed2[:,ii])
	mu1 = abs(G1)/p.sqrt(Rf*R1)
	mu2 = abs(G2)/p.sqrt(Rf*R2)

	# Get or recover axis and figure
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	x = np.arange(1,mu1.shape[0]+1) if f is None else gpu_to_cpu(f)
	mu1, mu2 = gpu_to_cpu(mu1), gpu_to_cpu(mu2)
	ax.plot(x, mu1, 'o-', color='blue', label='Reduced 1')
	ax.plot(x, mu2, 'o-', color='red', label='Reduced 2')
	ax.set_xlabel('St' if f is not None else 'Mode')
	ax.set_ylabel(r'$\mu$')
	ax.legend()
	return fig, ax