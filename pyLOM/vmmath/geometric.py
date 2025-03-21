#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - geometry.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp
from ..utils     import cr_nvtx as cr, mpi_reduce


@cr('math.cellCenters')
def cellCenters(xyz:np.ndarray,conec:np.ndarray) -> np.ndarray:
	r'''
	Compute the cell centers given a list 
	of elements.

	Args:
		xyz (np.ndarray):   node positions
		conec (np.ndarray): connectivity array

	Returns:
		np.ndarray: center positions
	'''
	p = cp if type(xyz) is cp.ndarray else np
	xyz_cen = p.zeros((conec.shape[0],xyz.shape[1]),xyz.dtype)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c = conec[ielem,conec[ielem,:]>=0]
		xyz_cen[ielem,:] = p.mean(xyz[c,:],axis=0)
	return xyz_cen

@cr('math.normals')
def normals(xyz:np.ndarray,conec:np.ndarray) -> np.ndarray:
	r'''
	Compute the cell normals given a list 
	of elements.

	Args:
		xyz (np.ndarray):   node positions
		conec (np.ndarray): connectivity array

	Returns:
		np.ndarray: cell normals
	'''
	p = cp if type(xyz) is cp.ndarray else np
	normals = p.zeros(((conec.shape[0],3)),xyz.dtype)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c     = conec[ielem,conec[ielem,:]>=0]
		xyzel =  xyz[c,:]
		# Compute centroid
		cen  = p.mean(xyzel,axis=0)
		# Compute normal
		for inod in range(len(c)):
			u = xyzel[inod]   - cen
			v = xyzel[inod-1] - cen
			normals[ielem,:] += 0.5*p.cross(u,v)
	return normals

@cr('math.euclidean_d')
def euclidean_d(X:np.ndarray) -> np.ndarray:
	r'''
	Compute the Euclidean distances between simulations.

	Args:
		X (np.ndarray): NxM Data matrix with N points in the mesh for M simulations

	Returns:
		np.ndarray: MxM distance matrix 
	'''
	p = cp if type(X) is cp.ndarray else np
	# Extract dimensions
	_,M = X.shape
	# Initialize distance matrix
	D = p.zeros((M,M),X.dtype)
	for i in range(M):
		for j in range(i+1,M,1):
			# Local sum on the partition
			d2 = p.sum((X[:,i]-X[:,j])*(X[:,i]-X[:,j]))
			# Global sum over the partitions
			dG = p.sqrt(mpi_reduce(d2,all=True))
			# Fill output
			D[i,j] = dG
			D[j,i] = dG
	# Return the mdistance matrix
	return D
