#!/usr/bin/env python
#
# pyLOM, utils.
#
# Mesh utilities
#
# Last rev: 20/10/2021
from __future__ import print_function, division

import numpy as np

from .errors import raiseError


STRUCT2D = ['structured2d','structured 2d','struct 2d','struct2d','s2d']
STRUCT3D = ['structured3d','structured 3d','struct 3d','struct3d','s3d']
UNSTRUCT = ['unstructured','unstr']

ELTYPE2ENSI = {
	'TRI03' : 'tria3',
	'QUA04' : 'quad4',
	'TET04' : 'tetra4',
	'PEN06' : 'penta6',
	'HEX08' : 'hexa8'
}
ELTYPE2VTK = {
}


def mesh_number_of_points(point,meshDict):
	'''
	Return number of points for a mesh
	'''
	if meshDict['type'].lower() in STRUCT2D:
		return meshDict['nx']*meshDict['ny'] if point else (meshDict['nx']-1)*(meshDict['ny']-1)
	if meshDict['type'].lower() in STRUCT3D:
		return meshDict['nx']*meshDict['ny']*meshDict['nz'] if point else (meshDict['nx']-1)*(meshDict['ny']-1)*(meshDict['nz']-1)
	if meshDict['type'].lower() in UNSTRUCT:
		return meshDict['nnod'] if point else meshDict['nel']
	raiseError('mesh type <%s> not recognized!'%meshDict['type'])


def mesh_element_type(meshDict,fmt):
	'''
	Return the type of element of the mesh
	'''
	if meshDict['type'].lower() in STRUCT2D: return 'quad4'
	if meshDict['type'].lower() in STRUCT3D: return 'hexa8'
	if meshDict['type'].lower() in UNSTRUCT: return ELTYPE2VTK[meshDict['eltype']] if 'vtk' in fmt else ELTYPE2ENSI[meshDict['eltype']]


def mesh_compute_connectivity(xyz,meshDict):
	'''
	Compute the connectivity array for structured meshes and return
	the connectivity for unstructured ones.
	'''
	# Connectivity for a 2D mesh
	if meshDict['type'].lower() in STRUCT2D: 
		nx, ny = meshDict['nx'], meshDict['ny']
		# Obtain the ids
		idx  = np.lexsort((xyz[:,1],xyz[:,0]))
		idx2 = idx.reshape((nx,ny))
		# Create connectivity array
		conec = np.zeros(((nx-1)*(ny-1),4),dtype=np.int32)
		conec[:,0] = idx2[:-1,:-1].ravel()
		conec[:,1] = idx2[:-1,1:].ravel()
		conec[:,2] = idx2[1:,1:].ravel()
		conec[:,3] = idx2[1:,:-1].ravel()
		conec     += 1 # Python index start at 0
	# Connectivity for a 3D mesh
	if meshDict['type'].lower() in STRUCT3D: 
		nx, ny, nz = meshDict['nx'], meshDict['ny'], meshDict['nz']
		# Obtain the ids
		idx  = np.lexsort((xyz[:,2],xyz[:,1],xyz[:,0]))
		idx2 = idx.reshape((nx,ny,nz))
		# Create connectivity array
		conec = np.zeros(((nx-1)*(ny-1)*(nz-1),8),dtype=np.int32)
		conec[:,0] = idx2[:-1,:-1,:-1].ravel()
		conec[:,1] = idx2[:-1,:-1,1:].ravel()
		conec[:,2] = idx2[:-1,1:,1:].ravel()
		conec[:,3] = idx2[:-1,1:,:-1].ravel()
		conec[:,4] = idx2[1:,:-1,:-1].ravel()
		conec[:,5] = idx2[1:,:-1,1:].ravel()
		conec[:,6] = idx2[1:,1:,:-1].ravel()
		conec[:,7] = idx2[1:,1:,1:].ravel()
		conec     += 1 # Python index start at 0
	# Connectivity for a unstructured mesh
	if meshDict['type'].lower() in UNSTRUCT: 
		idx   = np.arange(meshDict['nnod'],dtype=np.int32)
		conec = meshDict['conec']
	return conec, idx


def mesh_compute_cellcenter(xyz,meshDict):
	'''
	Compute cell centers given the node positions
	'''
	if meshDict['type'].lower() in STRUCT2D:
		nx, ny = meshDict['nx']-1, meshDict['ny']-1
		# Recover unique X, Y coordinates
		x = np.unique(xyz[:,0])
		y = np.unique(xyz[:,1])
		# Compute cell centers
		xc = x[:-1] + np.diff(x)/2.
		yc = y[:-1] + np.diff(y)/2.
		# Build xyzc
		xx, yy    = np.meshgrid(xc,yc,indexing='ij')
		xyzc      = np.zeros((nx*ny,2),dtype=np.double)
		xyzc[:,0] = xx.reshape((nx*ny,),order='C')
		xyzc[:,1] = yy.reshape((nx*ny,),order='C')
	# Connectivity for a 3D mesh
	if meshDict['type'].lower() in STRUCT3D:
		nx, ny, nz = meshDict['nx']-1, meshDict['ny']-1, meshDict['nz']-1
		# Recover unique X, Y coordinates
		x = np.unique(xyz[:,0])
		y = np.unique(xyz[:,1])
		z = np.unique(xyz[:,2])
		# Compute cell centers
		xc = x[:-1] + np.diff(x)/2.
		yc = y[:-1] + np.diff(y)/2.
		zc = z[:-1] + np.diff(z)/2.
		# Build xyzc
		xx, yy, zz = np.meshgrid(xc,yc,zc,indexing='ij')
		xyzc       = np.zeros((nx*ny*nz,3),dtype=np.double)
		xyzc[:,0]  = xx.reshape((nx*ny*nz,),order='C')
		xyzc[:,1]  = yy.reshape((nx*ny*nz,),order='C')		
		xyzc[:,2]  = zz.reshape((nx*ny*nz,),order='C')		
	# Connectivity for a unstructured mesh
	if meshDict['type'].lower() in UNSTRUCT:
		raiseError('Not yet implemented!')
	return xyzc


def mesh_reshape_var(var,meshDict,info):
	'''
	Reshape a variable according to the mesh
	'''
	# Obtain number of points from the mesh
	npoints = 0
	if meshDict['type'].lower() in STRUCT2D:
		npoints = meshDict['nx']*meshDict['ny'] if info['point'] else (meshDict['nx']-1)*(meshDict['ny']-1)
	if meshDict['type'].lower() in STRUCT3D:
		npoints = meshDict['nx']*meshDict['ny']*meshDict['nz']  if info['point'] else (meshDict['nx']-1)*(meshDict['ny']-1)*(meshDict['nz']-1)
	if meshDict['type'].lower() in UNSTRUCT:
		npoints = meshDict['nnod'] if info['point'] else meshDict['nel']
	# Only reshape the variable if ndim > 1
	out = np.ascontiguousarray(var.reshape((npoints,info['ndim']),order='F') if info['ndim'] > 1 else var)
	# Build 3D vector in case of 2D array
	if meshDict['type'].lower() in STRUCT2D and info['ndim'] == 2:
		out = np.hstack((out,np.zeros((npoints,1))))
	return out