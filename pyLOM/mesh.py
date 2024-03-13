#!/usr/bin/env python
#
# pyLOM, mesh.
#
# Mesh class, to organize data.
#
# Last rev: 26/01/2023
from __future__ import print_function, division

import numpy as np

from .vmmath       import cellCenters
from .utils.cr     import cr
from .utils.mem    import mem
from .utils.errors import raiseError
from .utils.parall import mpi_reduce


ALYA2ELTYP = {
	# Linear cells
	-1 :  0, # Empty cell
	 3 :  1, # Line element
	10 :  2, # Triangular cell
	12 :  3, # Quadrangular cell
	30 :  4, # Tetrahedral cell
	37 :  5, # Hexahedron
	34 :  6, # Linear prism
	32 :  7, # Pyramid
	# Quadratic, isoparametric cells
	39 : 15, # HEX27
	# Lagrangian cells
	40 : 25, # HEX64
}

ELTYPE2VTK = {
	 0 : 0,  # Empty cell
	 2 : 5,  # Triangular cell
	 3 : 9,  # Quadrangular cell
	 4 : 10, # Tetrahedral cell
	 6 : 13, # Linear prism
	 7 : 14, # Pyramid
	 5 : 12, # Hexahedron
	25 : 72, # Lagrangian Hexahedron
}

ELTYPE2ENSI = {
	2 : 'tria3',  # Triangular cell
	3 : 'quad4',  # Quadrangular cell
	4 : 'tetra4', # Tetrahedral cell
	6 : 'penta6', # Linear prism
	5 : 'hexa8',  # Hexahedron
}

MTYPE2ID = {
	'STRUCT2D' : 1,
	'STRUCT3D' : 2,
	'UNSTRUCT' : 3,
}

ID2MTYPE = {
	1 : 'STRUCT2D',
	2 : 'STRUCT3D',
	3 : 'UNSTRUCT',
}

class Mesh(object):
	'''
	The Mesh class wraps the mesh details of the case.
	'''
	@mem('Mesh')
	def __init__(self,mtype,xyz,connectivity,eltype,cellOrder,pointOrder):
		'''
		Class constructor
		'''
		self._type   = mtype
		self._xyz    = xyz
		self._xyzc   = None
		self._conec  = connectivity
		self._eltype = eltype
		self._cellO  = cellOrder
		self._pointO = pointOrder

	def __str__(self):
		'''
		String representation
		'''
		s   = 'Mesh (%s) of %d nodes and %d elements:\n' % (self.type,self.npoints,self.ncells)
		s  += '  > xyz  - max = ' + str(np.nanmax(self._xyz,axis=0)) + ', min = ' + str(np.nanmin(self._xyz,axis=0)) + '\n'
		return s

	def find_point(self,xyz):
		'''
		Return all the points where self._xyz == xyz
		'''
		return np.where(np.all(self._xyz == xyz,axis=1))[0]

	def find_cell(self,eltype):
		'''
		Return all the elements where self._elemList == elem
		'''
		return np.where(np.all(self._eltype == eltype))[0]

	def find_point_in_cell(self,inode):
		'''
		Return all the elements where the node is
		'''
		return np.where(np.any(np.isin(self._conec,inode),axis=1))[0]

	def size(self,pointData):
		'''
		Return the size accoding to the type of data
		'''
		return self.npoints if pointData else self.ncells

	@cr('Mesh.cellcenters')
	def cellcenters(self):
		'''
		Computes and returns the cell centers
		'''
		if self.type == 'STRUCT2D':
			# Recover unique X, Y coordinates
			x = np.unique(self.x)
			y = np.unique(self.y)
			# Compute cell centers
			xc = x[:-1] + np.diff(x)/2.
			yc = y[:-1] + np.diff(y)/2.
			# Build xyzc
			xx, yy    = np.meshgrid(xc,yc,indexing='ij')
			xyzc      = np.zeros((self.ncells,2),dtype=np.double)
			xyzc[:,0] = xx.reshape((self.ncells,),order='C')
			xyzc[:,1] = yy.reshape((self.ncells,),order='C')
		# Connectivity for a 3D mesh
		if self.type == 'STRUCT3D':
			# Recover unique X, Y, Z coordinates
			x = np.unique(self.x)
			y = np.unique(self.y)
			z = np.unique(self.z)
			# Compute cell centers
			xc = x[:-1] + np.diff(x)/2.
			yc = y[:-1] + np.diff(y)/2.
			zc = z[:-1] + np.diff(z)/2.
			# Build xyzc
			xx, yy, zz = np.meshgrid(xc,yc,zc,indexing='ij')
			xyzc       = np.zeros((self.ncells,3),dtype=np.double)
			xyzc[:,0]  = xx.reshape((self.ncells,),order='C')
			xyzc[:,1]  = yy.reshape((self.ncells,),order='C')		
			xyzc[:,2]  = zz.reshape((self.ncells,),order='C')		
		# Connectivity for a unstructured mesh
		if self.type == 'UNSTRUCT':
			xyzc = cellCenters(self._xyz,self._conec)
		return xyzc

	@cr('Mesh.reshape')
	def reshape_var(self,var,info):
		'''
		Reshape a variable according to the mesh
		'''
		# Obtain number of points from the mesh
		npoints = self.size(info['point'])
		# Only reshape the variable if ndim > 1
		out = np.ascontiguousarray(var.reshape((npoints,info['ndim']),order='C') if info['ndim'] > 1 else var)
		# Build 3D vector in case of 2D array
		if self.type == 'STRUCT2D' and info['ndim'] == 2:
			out = np.hstack((out,np.zeros((npoints,1))))
		return out

	@classmethod
	@cr('Mesh.new_struct2D')
	def new_struct2D(cls,nx,ny,x,y,dimsx,dimsy):
		xyz    = _struct2d_compute_xyz(nx,ny,x,y,dimsx,dimsy)
		conec  = _struct2d_compute_conec(nx,ny,xyz)
		eltype = 3*np.ones(((nx-1)*(ny-1),),np.uint8)
		cellO  = np.arange((nx-1)*(ny-1),dtype=np.int32)
		pointO = np.arange(nx*ny,dtype=np.int32)
		return cls('STRUCT2D',xyz,conec,eltype,cellO,pointO)

	@classmethod
	@cr('Mesh.new_struct3D')
	def new_struct3D(cls,nx,ny,nz,x,y,z,dimsx,dimsy,dimsz):
		xyz    = _struct3d_compute_xyz(nx,ny,nz,x,y,z,dimsx,dimsy,dimsz)
		conec  = _struct3d_compute_conec(nx,ny,nz,xyz)
		eltype = 5*np.ones(((nx-1)*(ny-1)*(nz-1),),np.uint8)
		cellO  = np.arange((nx-1)*(ny-1)*(nz-1),dtype=np.int32)
		pointO = np.arange(nx*ny*nz,dtype=np.int32)
		return cls('STRUCT3D',xyz,conec,eltype,cellO,pointO)

	@classmethod
	@cr('Mesh.from_pyAlya')
	def from_pyAlya(cls,mesh,sod=False):
		'''
		Create the mesh structure from a pyAlya mesh structure
		'''
		eltype = np.array([ALYA2ELTYP[t] for t in mesh.eltype],np.uint8)
		return cls('UNSTRUCT',mesh.xyz,mesh.connectivity_vtk if sod else mesh.connectivity,eltype,mesh.leinv,mesh.lninv)

	@property
	def type(self):
		return self._type
	@property
	def npoints(self):
		return self._xyz.shape[0]
	@property
	def npointsG(self):
		return mpi_reduce(self.npoints,op='sum',all=True)
	@property
	def npointsG2(self):
		return mpi_reduce(self.pointOrder.max(),op='max',all=True) + 1
	@property
	def ndim(self):
		return self._xyz.shape[1]
	@property
	def ncells(self):
		return self._eltype.shape[0]
	@property
	def ncellsG(self):
		return mpi_reduce(self.ncells,op='sum',all=True)
	@property
	def ncellsG2(self):
		return mpi_reduce(self.cellOrder.max(),op='max',all=True) + 1
	@property
	def nnodcell(self):
		return self._conec.shape[1]

	@property
	def xyz(self):
		return self._xyz
	@property
	def x(self):
		return self._xyz[:,0]
	@property
	def y(self):
		return self._xyz[:,1]
	@property
	def z(self):
		return self._xyz[:,2]
	@property
	def xyzc(self):
		if self._xyzc is None: self._xyzc = self.cellcenters()
		return self._xyzc

	@property
	def connectivity(self):
		return self._conec
	@property
	def cellOrder(self):
		return self._cellO
	@property
	def pointOrder(self):
		return self._pointO

	@property
	def eltype(self):
		return self._eltype
	@property
	def eltype2VTK(self):
		return np.array([ELTYPE2VTK[t] for t in self._eltype],np.uint8)
	@property
	def eltype2ENSI(self):
		return ELTYPE2ENSI[self._eltype[0]]


def _struct2d_compute_xyz(nx,ny,x,y,dimsx,dimsy):
	'''
	Compute points for a 2D structured mesh
	'''
	if x is None:
		dx = (dimsx[1] - dimsx[0])/(nx - 1.)
		x  = dx*np.arange(nx) + dimsx[0]
	if y is None:
		dy = (dimsy[1] - dimsy[0])/(ny - 1.)
		y  = dy*np.arange(ny) + dimsy[0]
	xx, yy = np.meshgrid(x,y,indexing='ij')
	xy = np.zeros((nx*ny,3),dtype=np.double)
	xy[:,0] = xx.reshape((nx*ny,),order='C')
	xy[:,1] = yy.reshape((nx*ny,),order='C')
	return xy

def _struct2d_compute_conec(nx,ny,xyz):
	'''
	Compute connectivity for a 2D structured mesh
	'''
	# Obtain the ids
	idx  = np.lexsort((xyz[:,1],xyz[:,0]))
	idx2 = idx.reshape((nx,ny))
	# Create connectivity array
	conec = np.zeros(((nx-1)*(ny-1),4),dtype=np.int32)
	conec[:,0] = idx2[:-1,:-1].ravel()
	conec[:,1] = idx2[:-1,1:].ravel()
	conec[:,2] = idx2[1:,1:].ravel()
	conec[:,3] = idx2[1:,:-1].ravel()
	return conec


def _struct3d_compute_xyz(nx,ny,nz,x,y,z,dimsx,dimsy,dimsz):
	'''
	Compute points for a 2D structured mesh
	'''
	if x is None:
		dx = (dimsx[1] - dimsx[0])/(nx - 1.)
		x  = dx*np.arange(nx) + dimsx[0]
	if y is None:
		dy = (dimsy[1] - dimsy[0])/(ny - 1.)
		y  = dy*np.arange(ny) + dimsy[0]
	if z is None:
		dz = (dimsz[1] - dimsz[0])/(nz - 1.)
		z  = dz*np.arange(nz) + dimsz[0]
	xx, yy, zz = np.meshgrid(x,y,z,indexing='ij')
	xyz = np.zeros((nx*ny*nz,3),dtype=np.double)
	xyz[:,0] = xx.reshape((nx*ny*nz,),order='C')
	xyz[:,1] = yy.reshape((nx*ny*nz,),order='C')
	xyz[:,2] = zz.reshape((nx*ny*nz,),order='C')
	return xyz

def _struct3d_compute_conec(nx,ny,nz,xyz):
	'''
	Compute connectivity for a 2D structured mesh
	'''
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
	conec[:,6] = idx2[1:,1:,1:].ravel()
	conec[:,7] = idx2[1:,1:,:-1].ravel()
	return conec
