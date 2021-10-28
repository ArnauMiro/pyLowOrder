#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Plotting utilities.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..vmmath     import vector_norm
from ..utils.mesh import mesh_compute_cellcenter


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
	return ax.contourf(X,Y,Z[ndim,:,:] if dim >= 0 else np.linalg.norm(Z,axis=0) if ndim > 1 else Z,cmap=cmap)


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
	accumulative_S  = np.array([vector_norm(S,ii) for ii in range(S.shape[0])],np.double)
	accumulative_S /= accumulative_S[0]
	# Plot accumulated residual
	ax.semilogy(np.arange(1,S.shape[0]+1),accumulative_S,'bo')
	# Set labels
	ax.set_ylabel(r'$\varepsilon_1$')
	ax.set_xlabel(r'Truncation size')
	ax.set_title(r'Tolerance')
	# Return
	return fig, ax

def plotSnapshot(X,xyz,mesh,info,dim=0,fig=None,ax=None,cmap=None):
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

def animateFlow(X,X_R,xyz,mesh,info,dim=0,t=None,fig=None,ax=None,cmap=None):
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
			plotFieldStruct2D(ax[1],mesh['nx'],mesh['ny'],info['ndim'],xyz,X_R[:,iframe],dim-1,cmap)
		else:
			plotFieldStruct2D(ax[0],mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,X[:,iframe],dim-1,cmap)
			plotFieldStruct2D(ax[1],mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,X_R[:,iframe],dim-1,cmap)
		ax[0].set_title('Real flow')
		ax[1].set_title('Reconstructed flow')
	anim = FuncAnimation(fig,update,frames=np.arange(nframes,dtype=np.int32),blit=False)
	return fig, ax, anim