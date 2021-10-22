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
from matplotlib.animation import FuncAnimation

from ..POD.wrapper import PSD
from ..utils.mesh  import mesh_compute_cellcenter


def plotFieldStruct2D(ax,nx,ny,ndim,xyz,field,dim,cmap,clear=False):
	'''
	Plot a 2D point or cell centered field on an structured mesh
	'''
	# Clear the axis if needed
	if clear: ax.clear()
	# Obtain the colormap if needed
	if cmap is None: cmap = plt.get_cmap('coolwarm',256)
	# Obtain X, Y matrices
	X = xyz[:,0].reshape((nx,ny),order='c').T
	Y = xyz[:,1].reshape((nx,ny),order='c').T
	# Obtain data matrix
	Z = field.reshape((nx,ny,ndim),order='c').T if ndim > 1 else field.reshape((nx,ny),order='c').T
	return ax.contourf(X,Y,Z[:,:,dim] if ndim > 1 else Z,cmap=cmap)


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

def plotMode(U,xyz,V,t,mesh,info,dim=1,modes=np.array([1],np.int32),scale_freq=1.,fig=[],ax=[],cmap=None):
	'''
	Given U, VT and a mode, plot their
	representation in a figure.
	'''
	cf = []
	for imode, mode in enumerate(modes):
		if len(fig) < imode + 1:
			fig.append( plt.figure(figsize=(8,6),dpi=100) )
		if len(ax) < imode + 1:
			ax.append( fig[imode].subplots(3,1,gridspec_kw = {'hspace':0.5}) )
		dt = t[1] - t[0]
		fig[imode].suptitle('Mode %d'%mode)
	   	# Plot the representation of mode U
		if mesh['type'] == 'struct2D':
			c = None
			if info['point']:
				c = plotFieldStruct2D(ax[imode][0],mesh['nx'],mesh['ny'],info['ndim'],xyz,U[:,mode-1],dim-1,cmap)
			else:
				xyzc = mesh_compute_cellcenter(xyz,mesh)
				c = plotFieldStruct2D(ax[imode][0],mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,U[:,mode-1],dim-1,cmap)
			cf.append(c)
		ax[imode][0].set_title('Spatial mode')
	   	# Plot the temporal evolution of the mode
		ax[imode][1].plot(t,V[mode-1,:],'b')
		ax[imode][1].set_title('Temporal mode')
		# Plot frequency representation of the mode
		p, freq = PSD(V,dt,m=mode)
		freq *= scale_freq
		L = int(np.floor(V[mode-1,:].shape[0]/2))
		ax[imode][2].plot(freq[:L],p[:L])
		ax[imode][2].set_title('Power Spectrum')
		ax[imode][2].set_xlabel('St')
	return fig, ax, cf

def plotDMDMode(U, xyz, mesh, omegas, modes=np.array([1],np.int32), fig=[],ax=[],cmap=None):
	'''
	Given U and a mode, plot its representation in a figure.
	'''
	cf = []
	for imode, mode in enumerate(modes):
		if len(fig) < imode + 1:
			fig.append( plt.figure(figsize=(8,6),dpi=100) )
		if len(ax) < imode + 1:
			ax.append( fig[imode].subplots(2,1,gridspec_kw = {'hspace':0.5}) )
		fig[imode].suptitle('Mode %d, f = %f [Hz]'%(mode, omegas[modes[imode] - 1]))
#		if mesh['type'] == 'struct2D':
#			cf.append(plotFieldStruct2D(ax[imode][0],mesh['nx'],mesh['ny'],xyz,np.real(U[:,mode-1]),cmap) )
#			cf.append(plotFieldStruct2D(ax[imode][1],mesh['nx'],mesh['ny'],xyz,np.imag(U[:,mode-1]),cmap) )
		ax[imode][0].set_title('Real part of the mode')
		ax[imode][1].set_title('Imaginary part of the mode')
	return fig, ax, cf


def plotSnapshot(X,xyz,mesh,info,dim=1,fig=None,ax=None,cmap=None):
	'''
	Given X and the mesh plot a time instant
	'''
	# Build figure and axes
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.add_subplot(1,1,1)
	# Contour plot
	cf = None
	if info['point']:
		cf = plotFieldStruct2D(ax,mesh['nx'],mesh['ny'],info['ndim'],xyz,X,dim-1,cmap)
	else:
		xyzc = mesh_compute_cellcenter(xyz,mesh)
		cf = plotFieldStruct2D(ax,mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,X,dim-1,cmap)
	return fig, ax, cf


def animateFlow(X,X_POD,xyz,mesh,info,dim=1,t=None,fig=None,ax=None,cmap=None):
	'''
	Given X and the mesh plot a time instant
	'''
	# Build figure and axes
	if fig is None:
		fig = plt.figure(figsize=(8,6),dpi=100)
	if ax is None:
		ax = fig.subplots(2,1,gridspec_kw = {'hspace':0.5})
    	# Select frames to animate
	nframes = X.shape[1]
	xyzc = None if info['point'] else mesh_compute_cellcenter(xyz,mesh)
	# Function to animate
	def update(iframe):
		fig.suptitle('Snapshot no %d'%iframe)
		if info['point']:
			plotFieldStruct2D(ax[0],mesh['nx'],mesh['ny'],info['ndim'],xyz,X[:,iframe],dim-1,cmap)
			plotFieldStruct2D(ax[1],mesh['nx'],mesh['ny'],info['ndim'],xyz,X_POD[:,iframe],dim-1,cmap)
		else:
			plotFieldStruct2D(ax[0],mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,X[:,iframe],dim-1,cmap)
			plotFieldStruct2D(ax[1],mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,X_POD[:,iframe],dim-1,cmap)			
		ax[0].set_title('Real flow')
		ax[1].set_title('Reconstructed flow')
	anim = FuncAnimation(fig,update,frames=np.arange(nframes,dtype=np.int32),blit=False)
	return fig, ax, anim
