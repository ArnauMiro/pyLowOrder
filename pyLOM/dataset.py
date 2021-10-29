#!/usr/bin/env python
#
# pyLOM, dataset.
#
# Dataset class, reader and reduction routines.
#
# Last rev: 30/07/2021
from __future__ import print_function, division

import os, copy, mpi4py, numpy as np
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from .             import inp_out as io
from .utils.cr     import cr_start, cr_stop
from .utils.errors import raiseError
from .utils.mesh   import mesh_number_of_points, mesh_reshape_var, mesh_element_type, mesh_compute_connectivity, mesh_compute_cellcenter


POS_KEYS  = ['xyz','coords','pos']
TIME_KEYS = ['time']


class Dataset(object):
	'''
	The Dataset class wraps the position of the nodes and the time instants
	with the number of variables and relates them so that the operations 
	in parallel are easier.
	'''
	def __init__(self, mesh=None, xyz=np.array([[0.,0.,0.]],np.double), time=np.array([0.],np.double), 
		pointOrder=np.array([],np.int32), cellOrder=np.array([],np.int32), **kwargs):
		'''
		Class constructor (self, mesh, xyz, time, **kwargs)

		Inputs:
			> mesh:  dictionary containing the mesh details.
			> xyz:   position of the nodes as a numpy array of 3 dimensions.
			> time:  time instants as a numpy array.
			> kwags: dictionary containin the variable name and values as a
					 python dictionary.
		'''
		npoints          = mesh_number_of_points(True,mesh)
		ncells           = mesh_number_of_points(False,mesh)
		self._xyz        = xyz
		self._xyzc       = None
		self._time       = time
		self._pointOrder = np.arange(npoints,dtype=np.int32) if len(pointOrder) == 0 else pointOrder
		self._cellOrder  = np.arange(npoints,dtype=np.int32) if len(cellOrder)  == 0 else cellOrder
		self._vardict    = kwargs
		self._meshDict   = mesh

	def __len__(self):
		return self._xyz.shape[0]

	def __str__(self):
		'''
		String representation
		'''
		shp = (self._xyz.shape[0], self._time.shape[0])
		s   = 'Dataset of %d elements and %d instants:\n' % (shp[0],shp[1])
		s  += '  > xyz  - max = ' + str(np.nanmax(self._xyz,axis=0)) + ', min = ' + str(np.nanmin(self._xyz,axis=0)) + '\n'
		s  += '  > time - max = ' + str(np.nanmax(self._time,axis=0)) + ', min = ' + str(np.nanmin(self._time,axis=0)) + '\n'
		for key in self.varnames:
			var = self[key]
			nanstr = ' (has NaNs) ' if np.any(np.isnan(var)) else ' '
			s  += '  > ' +  key + nanstr + '- max = ' + str(np.nanmax(var)) \
										 + ', min = ' + str(np.nanmin(var)) \
										 + ', avg = ' + str(np.nanmean(var)) \
										 + '\n'
		return s
		
	# Set and get functions
	def __getitem__(self,key):
		'''
		Dataset[key]

		Recover the value of a variable given its key
		'''
		if key in POS_KEYS:
			return self._xyz
		elif key in TIME_KEYS:
			return self._time
		else:
			return self._vardict[key]['value']

	def __setitem__(self,key,value):
		'''
		Dataset[key] = value

		Set the value of a variable given its key
		'''
		if key in POS_KEYS:
			self._xyz = value
		elif key in TIME_KEYS:
			self._time = value
		else:
			self._vardict[key] = value

	# Functions
	def find(self,xyz):
		'''
		Return all the points where self._xyz == xyz
		'''
		return np.where(np.all(self._xyz == xyz,axis=1))[0]

	def rename(self,new,old):
		'''
		Rename a variable inside a field.
		'''
		self.var[new] = self.var.pop(old)
		return self

	def delete(self,varname):
		'''
		Delete a variable inside a field.
		'''
		return self.var.pop(varname)

	def info(self,var):
		'''
		Returns the information for a certain variable
		'''
		return {'point':self._vardict[var]['point'],'ndim':self._vardict[var]['ndim']}

	def add_variable(self,varname,point,ndim,ninst,var):
		'''
		Add a variable to the dataset
		'''
		self._vardict[varname] = {
			'point' : point,
			'ndim'  : ndim,
			'value' : var, 
		}

	def cellcenters(self):
		'''
		Computes and returns the cell centers
		'''
		return mesh_compute_cellcenter(self._xyz,self._meshDict)

	def X(self,*args):
		'''
		Return the X matrix for the selected variables
		'''
		# Select all variables if none is provided
		variables = self.varnames if len(args) == 0 else args
		# Compute the number of variables
		nvars = 0
		for var in variables:
			nvars += self.var[var]['ndim']
		# Create output array
		npoints = self.pointOrder.shape[0] if self.var[variables[0]]['point'] else self.cellOrder.shape[0]
		ninst   = self._time.shape[0]
		X = np.zeros((nvars*npoints,ninst),np.double)
		# Populate output matrix
		ivar = 0
		for var in variables:
			v = self.var[var]
			if v['ndim'] == 1:
				X[ivar:nvars*npoints:nvars] = v['value']
				ivar += 1
			else:
				for idim in range(v['ndim']):
					X[ivar:nvars*npoints:nvars] = v['value'][:,idim]
					ivar += 1
		return X

	def save(self,fname,**kwargs):
		'''
		Store the field in various formats.
		'''
		cr_start('Dataset.save',0)
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		# Pickle format
		if fmt.lower() == 'pkl': 
			io.pkl_save(fname,self)
		# H5 format
		if fmt.lower() == 'h5':
			# Set default parameters
			if not 'mpio'         in kwargs.keys(): kwargs['mpio']         = True
			if not 'write_master' in kwargs.keys(): kwargs['write_master'] = False
			io.h5_save(fname,self.xyz,self.time,self._pointOrder,self._cellOrder,self.mesh,self.var,**kwargs)
		cr_stop('Dataset.save',0)

	@classmethod
	def load(cls,fname,**kwargs):
		'''
		Load a field from various formats
		'''
		cr_start('Dataset.load',0)
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		# Pickle format
		if fmt.lower() == 'pkl': 
			cr_stop('Dataset.load',0)
			return io.pkl_load(fname)
		# H5 format
		if fmt.lower() == 'h5':
			if not 'mpio' in kwargs.keys(): kwargs['mpio'] = True
			xyz, time, pointOrder, cellOrder, meshDict, varDict = io.h5_load(fname,**kwargs)
			cr_stop('Dataset.load',0)
			return cls(meshDict,xyz,time,pointOrder,cellOrder,**varDict)
		cr_stop('Dataset.load',0)
		raiseError('Cannot load file <%s>!'%fname)

	def write(self,casestr,basedir='./',instants=[0],vars=[],fmt='vtk'):
		'''
		Store the data using various formats.
		This method differs from save in the fact that save is used 
		to recover the field, write only outputs the data.
		'''
		cr_start('Dataset.write',0)
		if fmt.lower() in ['vtk']:
			cr_stop('Dataset.write',0)
			raiseError('VTK format not yet implemented!')
		elif fmt.lower() in ['ensi','ensight']:
			EnsightWriter(self,casestr,basedir,instants,vars)
		else:
			cr_stop('Dataset.write',0)
			raiseError('Format <%s> not implemented!'%fmt)
		cr_stop('Dataset.write',0)

	# Properties
	@property
	def xyz(self):
		return self._xyz
	@xyz.setter
	def xyz(self,value):
		self._xyz = value
	@property
	def xyzc(self):
		if self._xyzc is None: self._xyzc = self.cellcenters()
		return self._xyzc

	@property
	def time(self):
		return self._time
	@time.setter
	def time(self,value):
		self._time = value

	@property
	def pointOrder(self):
		return self._pointOrder
	@property
	def cellOrder(self):
		return self._cellOrder

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
	def mesh(self):
		return self._meshDict
	@mesh.setter
	def mesh(self,value):
		self._meshDict = value

	@property
	def var(self):
		return self._vardict
	@property
	def varnames(self):
		return list(self._vardict.keys())


def EnsightWriter(dset,casestr,basedir,instants,varnames):
	'''
	Ensight dataset writer
	'''
	# Create the filename for the geometry
	geofile = os.path.join(basedir,'%s.ensi.geo'%casestr)
	header = {
		'descr'  : 'File created with pyAlya tool\nmesh file',
		'nodeID' : 'assign',
		'elemID' : 'assign',
		'partID' : 1,
		'partNM' : 'Volume Mesh',
		'eltype' : mesh_element_type(dset.mesh,'ensi')
	}
	# Write geometry file
	conec = mesh_compute_connectivity(dset.xyz,dset.mesh)
	io.Ensight_writeGeo(geofile,dset.xyz,conec,header)
	# Write instantaneous fields
	binfile_fmt = '%s.ensi.%s-%06d'
	# Define Ensight header
	header = {
		'descr'  : 'File created with pyLOM',
		'partID' : 1,
		'partNM' : 'part',
		'eltype' : mesh_element_type(dset.mesh,'ensi')
	}
	# Loop the selected instants
	for var in varnames:
		# Recover variable information
		info  = dset.info(var)
		field = dset[var]
		# Variable has temporal evolution
		header['eltype'] = 'coordinates' if info['point'] else mesh_element_type(dset.mesh,'ensi')
		if len(field.shape) > 1:
			# Loop requested instants
			for instant in instants:
				filename = os.path.join(basedir,binfile_fmt % (casestr,var,instant+1))
				# Reshape variable for Ensight file
				f = mesh_reshape_var(field[:,instant],dset.mesh,info)
				io.Ensight_writeField(filename,f,header)
		else:
			filename = os.path.join(basedir,binfile_fmt % (casestr,var,1))
			# Reshape variable for Ensight file
			f = mesh_reshape_var(field,dset.mesh,info)
			io.Ensight_writeField(filename,f,header)