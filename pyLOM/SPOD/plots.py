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
from ..utils       import gpu_to_cpu, MPI_RANK, mpi_reduce
from ..utils.plots import plotLayout
from ..vmmath      import vecmat, vector_sum, vdot, abs
from ..utils.gpu   import cp
from ..            import Mesh, Dataset


def plotMode(L:np.ndarray, P:np.ndarray, freqs:np.ndarray, mesh:Mesh, dset:Dataset, ivar:int, pointData:bool=True, modes:np.ndarray=np.array([1],np.int32),**kwargs):
	r'''
	Plot a SPOD mode. For complex modes both the real and the imaginary parts
	are plotted side by side, for real modes a single field is plotted.

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
	# Extract the modes to be plotted. The fields must be real (the
	# visualization backends work with real scalars), so a complex mode
	# is shown as the pair of real fields (real, imaginary part)
	npoints = mesh.size(pointData)
	if np.iscomplexobj(P):
		dset.add_field('P_MODES_REAL',len(modes),extract_modes(L,P,ivar,npoints,modes=modes,kind='real'))
		dset.add_field('P_MODES_IMAG',len(modes),extract_modes(L,P,ivar,npoints,modes=modes,kind='imag'))
		vars, ncols, title = ['P_MODES_REAL','P_MODES_IMAG'], 2, 'Mode %d St = %.3f (real / imag)'
	else:
		dset.add_field('P_MODES',len(modes),extract_modes(L,P,ivar,npoints,modes=modes))
		vars, ncols, title = ['P_MODES'], 1, 'Mode %d St = %.3f'
	# Loop over the modes
	screenshot = kwargs.pop('screenshot',None)
	off_screen = kwargs.pop('off_screen',False)
	for imode, mode in enumerate(modes):
		if screenshot is not None: kwargs['screenshot'] = screenshot % imode
		plotLayout(mesh,dset,1,ncols,mode-1,vars=vars,title=title % (mode-1, np.abs(freqs[mode-1])),off_screen=off_screen,**kwargs)
	# Remove from dataset
	for v in vars: dset.delete(v)

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
	the same nDFT, nolap and dt so that the frequencies match. For complex modes they must
	be extracted with kind='complex' so that the phase information is kept. The y axis runs
	from 0 to 1 with a small headroom above 1 so that the fully converged modes are visible,
	and a dashed reference line is drawn at mu = 0.95.

	For real modes, mu is the absolute value of the cosine of the angle between the
	reference and the reduced modes. For complex modes, mu is the phase-matched cosine,
	i. e. the modulus of the W-weighted complex inner product normalized by the norms,
	which is insensitive to the arbitrary phase (gauge) that each SPOD run assigns
	to its modes.

	.. note::
	As the spatial modes are distributed over the processors, the per-mode sums are
	computed on the local slice and reduced over the whole domain before the plot is
	drawn. The reduction is collective, so every rank must call this function; only
	the root rank produces the figure.

	Args:
		PRef (np.ndarray): reference modes, (npoints, nmodes) array as returned by extract_modes with reshape=False
		PRed1 (np.ndarray): modes of the first reduced dataset, (npoints, nmodes) array as returned by extract_modes with reshape=False
		PRed2 (np.ndarray): modes of the second reduced dataset, (npoints, nmodes) array as returned by extract_modes with reshape=False
		W (np.ndarray, optional): diagonal of the inner product weighting matrix (default: ``None``)
		f (np.ndarray, optional): frequencies corresponding to the columns of the modes, used as x axis and sorted in increasing order before plotting (default: ``None``, i. e. mode id)
		fig (plt.figure, optional): figure object in which the plot will be done (default: ``[]``)
		ax (plt.axes, optional): axes object in which the plot will be done (default: ``[]``)

	Returns:
		[plt.figure, plt.axes]: figure and axes objects of the plot on the root rank, ``None`` on the others
	'''
	p = cp if type(PRef) is cp.ndarray else np
	cplx = np.iscomplexobj(PRef)
	if W is not None:
		# vecmat shares a single dtype between the vector and the matrix,
		# so a real weight vector and the complex modes are scaled
		# elementwise instead
		W = p.sqrt(W)
		if cplx:
			PRef  = W[:,None]*PRef
			PRed1 = W[:,None]*PRed1
			PRed2 = W[:,None]*PRed2
		else:
			PRef  = vecmat(W, PRef)
			PRed1 = vecmat(W, PRed1)
			PRed2 = vecmat(W, PRed2)

	# Each rank only holds the local slice of the modes, so the per-mode
	# sums are computed locally and reduced over the whole domain below.
	# For complex modes the inner product is the phase-matched one, sum of
	# Ref*conj(Red): as vdot conjugates its first argument, the reduced
	# mode is passed first.
	ndtype = PRef.dtype if cplx else None
	nmodes = PRef.shape[1]
	G1 = p.zeros(nmodes, dtype=ndtype) 
	G2 = p.zeros(nmodes, dtype=ndtype)
	Rf = p.zeros(nmodes, dtype=ndtype) 
	R1 = p.zeros(nmodes, dtype=ndtype)
	R2 = p.zeros(nmodes, dtype=ndtype)
	for ii in range(nmodes):
		if cplx:
			G1[ii] = vdot(PRed1[:,ii], PRef[:,ii])
			G2[ii] = vdot(PRed2[:,ii], PRef[:,ii])
			Rf[ii] = vdot(PRef[:,ii], PRef[:,ii])
			R1[ii] = vdot(PRed1[:,ii], PRed1[:,ii])
			R2[ii] = vdot(PRed2[:,ii], PRed2[:,ii])
		else:
			G1[ii] = vector_sum(PRef[:,ii]  * PRed1[:,ii])
			G2[ii] = vector_sum(PRef[:,ii]  * PRed2[:,ii])
			Rf[ii] = vector_sum(PRef[:,ii]  * PRef[:,ii])
			R1[ii] = vector_sum(PRed1[:,ii] * PRed1[:,ii])
			R2[ii] = vector_sum(PRed2[:,ii] * PRed2[:,ii])
	# Reduce the local partial sums over the whole domain. This is a
	# collective call: every rank must reach this point. It is a no-op
	# when running serially, so the serial results are unchanged.
	sums = mpi_reduce(gpu_to_cpu(p.stack((G1,G2,Rf,R1,R2))), op='sum', all=True)
	G1, G2, Rf, R1, R2 = sums
	mu1 = (abs(G1)/np.sqrt(Rf*R1)).real
	mu2 = (abs(G2)/np.sqrt(Rf*R2)).real

	# Only the root rank draws the figure
	if MPI_RANK != 0:
		return fig, ax

	# Get or recover axis and figure
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	if f is not None:
		# Plot in increasing frequency order: the columns of the modes are
		# sorted by energy, not by frequency, and an unsorted x axis would
		# draw the curves as a set of crossing chords
		f = gpu_to_cpu(f)
		order = np.argsort(f)
		x, mu1, mu2 = f[order], mu1[order], mu2[order]
	else:
		x = np.arange(1,mu1.shape[0]+1)

	ax.axhline(0.95, color='k', linestyle='--', label=r'$\mu = 0.95$')
	# A little headroom above 1 so that the mu = 1 markers are not
	# clipped by the top border of the axis
	ax.set_ylim(0, 1.05)
	ax.plot(x, mu1, 'o-', color='blue', label='Mode 1')
	ax.plot(x, mu2, 'o-', color='red', label='Mode 2')
	ax.set_xlabel('St' if f is not None else 'Mode')
	ax.set_ylabel(r'$\mu$')
	ax.legend()
	return fig, ax