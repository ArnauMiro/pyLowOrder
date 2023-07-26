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

from ..vmmath       import vector_norm
from ..utils.cr     import cr
from ..utils.errors import raiseWarning


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
	levels = np.linspace(-1e-2, 1e-2, 11)
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


try:
	import pyvista as pv

	def _cells_and_offsets(conec):
		'''
		Build the offsets and cells array to create am
		UnstructuredGrid
		'''
		# Compute points per cell
		ppcell = np.sum(conec >= 0,axis=1)
		# Compute cells for pyVista, with the number of points per cell on front
		cells = np.c_[ppcell,conec]
		# Now we get rid of any -1 entries for mixed meshes
		cellsf = cells.flatten('c')
		cellsf = cellsf[cellsf>=0]
		# Now build the offsets vector
		offset = np.zeros((ppcell.shape[0]+1,),np.int32)
		offset[1:] = np.cumsum(ppcell)
		return cellsf, offset

	@cr('plots.pyvista_snap')
	def plotSnapshot(dset,vars=[],instant=0,**kwargs):
		'''
		Plot using pyVista
		'''
		mesh = dset.mesh
		# First create the unstructured grid
		cells, offsets = _cells_and_offsets(mesh.connectivity)
		# Create the unstructured grid
		ugrid =  pv.UnstructuredGrid(offsets,cells,mesh.eltype2VTK,mesh.xyz) if pv.vtk_version_info < (9,) else pv.UnstructuredGrid(cells,mesh.eltype2VTK,mesh.xyz)
		# Load the variables inside the unstructured grid
		for v in vars:
			info = dset.info(v)
			if info['point']:
				ugrid.point_data[v] = dset.mesh.reshape_var(dset[v][:,instant] if len(dset[v].shape) > 1 else dset[v],info)
			else:
				ugrid.cell_data[v]  = dset.mesh.reshape_var(dset[v][:,instant] if len(dset[v].shape) > 1 else dset[v],info)
		# Launch plot
		return ugrid.plot(**kwargs)

	@cr('plots.pyvista_layout')
	def plotLayout(dset,nrows,ncols,imode,vars=[],cmap='jet',title='',off_screen=False,**kwargs):
		'''
		Plot using pyVista
		'''
		mesh = dset.mesh
		# First create the unstructured grid
		cells, offsets = _cells_and_offsets(mesh.connectivity)
		# Create the unstructured grid
		ugrid =  pv.UnstructuredGrid(offsets,cells,mesh.eltype2VTK,mesh.xyz) if pv.vtk_version_info < (9,) else pv.UnstructuredGrid(cells,mesh.eltype2VTK,mesh.xyz)
		# Load the variables inside the unstructured grid
		plotter = pv.Plotter(shape=(nrows,ncols),off_screen=off_screen)
		irow, icol = 0, 0
		for ivar, v in enumerate(vars):
			# Add variable to mesh
			info = dset.info(v)
			if info['point']:
				ugrid.point_data[v] = dset.mesh.reshape_var(dset[v],info)
			else:
				ugrid.cell_data[v]  = dset.mesh.reshape_var(dset[v],info)
			# Plot
			plotter.subplot(irow,icol)
			if ivar == 0: plotter.add_text(title)
			plotter.add_mesh(ugrid,scalars=v,cmap=cmap,component=imode,copy_mesh=True)
			# Check irow, icol bounds
			icol += 1
			if icol == ncols:
				icol  = 0
				irow += 1
		# Launch plot
		return plotter.show(**kwargs)

except:
	def plotSnapshot(dset,vars=[],instant=0,**kwargs):
		'''
		Plot using pyVista
		'''
		raiseWarning('Import - Problems loading pyVista!',False)

	def plotLayout(Phi,mesh,nrows,ncols,vars=[],title='',**kwargs):
		'''
		Plot using pyVista
		'''
		raiseWarning('Import - Problems loading pyVista!',False)
