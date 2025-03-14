#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - geometry.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.cr     import cr
from ..utils.parall import mpi_reduce


@cr('math.cellCenters')
def cellCenters(xyz,conec):
	'''
	Compute the cell centers given a list 
	of elements.
	'''
	xyz_cen = np.zeros((conec.shape[0],xyz.shape[1]),xyz.dtype)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c = conec[ielem,conec[ielem,:]>=0]
		xyz_cen[ielem,:] = np.mean(xyz[c,:],axis=0)
	return xyz_cen

@cr('math.normals')
def normals(xyz,conec):
	normals = np.zeros(((conec.shape[0],3)),xyz.dtype)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c     = conec[ielem,conec[ielem,:]>=0]
		xyzel =  xyz[c,:]
		# Compute centroid
		cen  = np.mean(xyzel,axis=0)
		# Compute normal
		for inod in range(len(c)):
			u = xyzel[inod]   - cen
			v = xyzel[inod-1] - cen
			normals[ielem,:] += 0.5*np.cross(u,v)
	return normals

@cr('math.euclidean_d')
def euclidean_d(X):
	'''
	Compute Euclidean distances between simulations.

	In:
		- X: NxM Data matrix with N points in the mesh for M simulations
	Returns:
		- D: MxM distance matrix 
	'''
	# Extract dimensions
	_,M = X.shape
	# Initialize distance matrix
	D = np.zeros((M,M),X.dtype)
	for i in range(M):
		for j in range(i+1,M,1):
			# Local sum on the partition
			d2 = np.sum((X[:,i]-X[:,j])*(X[:,i]-X[:,j]))
			# Global sum over the partitions
			dG = np.sqrt(mpi_reduce(d2,all=True))
			# Fill output
			D[i,j] = dG
			D[j,i] = dG
	# Return the mdistance matrix
	return D
