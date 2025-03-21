#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - geometry.
#
# Last rev: 27/10/2021

cimport cython
cimport numpy as np

import numpy as np

from .cfuncs       cimport real
from .cfuncs       cimport c_scellCenters, c_snormals, c_seuclidean_d
from .cfuncs       cimport c_dcellCenters, c_dnormals, c_deuclidean_d

from ..utils.cr     import cr
from ..utils.errors import raiseError


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _scellCenters(float[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] xyz_cen = np.zeros((nel,ndim),dtype = np.float32)
	# Call C function
	c_scellCenters(&xyz_cen[0,0],&xyz[0,0],&conec[0,0],nel,ndim,ncon)
	# Return
	return xyz_cen

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dcellCenters(double[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] xyz_cen = np.zeros((nel,ndim),dtype = np.double)
	# Call C function
	c_dcellCenters(&xyz_cen[0,0],&xyz[0,0],&conec[0,0],nel,ndim,ncon)
	# Return
	return xyz_cen

@cr('math.cellCenters')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cellCenters(real[:,:] xyz, int[:,:] conec):
	r'''
	Compute the cell centers given a list 
	of elements.

	Args:
		xyz (np.ndarray):   node positions
		conec (np.ndarray): connectivity array

	Returns:
		np.ndarray: center positions
	'''
	if real is double:
		return _dcellCenters(xyz,conec)
	else:
		return _scellCenters(xyz,conec)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _snormals(float[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] normals = np.zeros((nel,ndim),dtype = np.float32)
	# Call C function
	c_snormals(&normals[0,0],&xyz[0,0],&conec[0,0],nel,ndim,ncon)
	# Return
	return normals

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dnormals(double[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] normals = np.zeros((nel,ndim),dtype = np.double)
	# Call C function
	c_dnormals(&normals[0,0],&xyz[0,0],&conec[0,0],nel,ndim,ncon)
	# Return
	return normals

@cr('math.normals')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def normals(real[:,:] xyz, int[:,:] conec):
	r'''
	Compute the cell normals given a list 
	of elements.

	Args:
		xyz (np.ndarray):   node positions
		conec (np.ndarray): connectivity array

	Returns:
		np.ndarray: cell normals
	'''
	if real is double:
		return _dnormals(xyz,conec)
	else:
		return _snormals(xyz,conec)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _seuclidean_d(float[:,:] X):
	'''
	Compute Euclidean distances between simulations.

	In:
		- X: NxM Data matrix with N points in the mesh for M simulations
	Returns:
		- D: MxM distance matrix 
	'''
	# Initialize
	cdef int n = X.shape[0], m = X.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] D = np.zeros((m,m),dtype=np.float32)
	# Call C function
	c_seuclidean_d(&D[0,0],&X[0,0],n,m);
	# Return the distance matrix
	return D

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _deuclidean_d(double[:,:] X):
	'''
	Compute Euclidean distances between simulations.

	In:
		- X: NxM Data matrix with N points in the mesh for M simulations
	Returns:
		- D: MxM distance matrix 
	'''
	# Initialize
	cdef int n = X.shape[0], m = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] D = np.zeros((m,m),dtype=np.double)
	# Call C function
	c_deuclidean_d(&D[0,0],&X[0,0],n,m);
	# Return the distance matrix
	return D

@cr('math.euclidean_d')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def euclidean_d(real[:,:] X):
	r'''
	Compute the Euclidean distances between simulations.

	Args:
		X (np.ndarray): NxM Data matrix with N points in the mesh for M simulations

	Returns:
		np.ndarray: MxM distance matrix 
	'''
	if real is double:
		return _deuclidean_d(X)
	else:
		return _seuclidean_d(X)