#!/usr/bin/env python
#
# pyLOM, dataset.
#
# Dataset class, reader and reduction routines.
#
# Last rev: 30/07/2021
from __future__ import print_function, division

import os, mpi4py, numpy as np
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from .                import inp_out as io
from .utils.cr        import cr
from .utils.mem       import mem
from .utils.errors    import raiseError
from .utils.parall    import mpi_reduce
from .partition_table import PartitionTable


class Dataset(object):
	'''
	The Dataset class wraps the position of the nodes and the time instants
	with the number of variables and relates them so that the operations 
	in parallel are easier.
	'''
	def __init__(self, xyz=None, ptable=None, vars=None, order=None, point=True, **kwargs):
		'''
		Class constructor

		Inputs:
			> xyz:    coordinates of the points.
			> ptable: partition table used.
			> vars:   dictionary containing the variable name and values as
					  as a python dictionary.
			> order:  ordering of the points (automatically created if none)
			> point:  True if point data, False if cell data.
			> kwags:  dictionary containing the field name and values as a
					  python dictionary.
		'''
		self._xyz      = xyz
		self._vardict  = vars
		self._fieldict = kwargs
		self._ptable   = ptable
		self._order    = np.arange(xyz.shape[0]) if order is None else order
		self._point    = point

	def __len__(self):
		return self._xyz.shape[0]

	def __str__(self):
		'''
		String representation
		'''
		s  = 'Dataset of %d variables:\n' % len(self.varnames)
		for key in self.varnames:
			var    = self.vars[key]['value']
			nanstr = ' (has NaNs) ' if np.any(np.isnan(var)) else ' '
			s     += '  > ' + key + nanstr + ' - max = ' + str(np.nanmax(var)) + ', min = ' + str(np.nanmin(var)) + '\n'
		s += 'and %d fields with %d points:\n' % (len(self.fieldnames),len(self))
		for key in self.fieldnames:
			field  = self.fields[key]['value']
			nanstr = ' (has NaNs) ' if np.any(np.isnan(field)) else ' '
			s     += '  > ' +  key + nanstr + '- max = ' + str(np.nanmax(field)) \
											+ ', min = ' + str(np.nanmin(field)) \
											+ ', avg = ' + str(np.nanmean(field)) \
											+ '\n'
		return s
		
	# Set and get functions
	def __getitem__(self,key):
		'''
		Dataset[key]

		Recover the value of a field given its key
		'''
		return self._fieldict[key]['value']

	def __setitem__(self,key,value):
		'''
		Dataset[key] = value

		Set the field of a variable given its key
		'''
		self._fieldict[key]['value'] = value

	# Functions
	def rename(self,new,old):
		'''
		Rename a variable inside a field.
		'''
		self.fields[new] = self.fields.pop(old)
		return self

	def delete(self,varname):
		'''
		Delete a variable inside a field.
		'''
		return self.fields.pop(varname)

	def get_variable(self,key):
		'''
		Recover the value of a variable given its key
		'''
		return self._vardict[key]['value']

	def set_variable(self,key,value):
		'''
		Recover the value of a variable given its key
		'''
		self._vardict[key]['value'] = value

	def get_dim(self,var,idim):
		'''
		Recover the value of a variable for a given dimension
		'''
		ndim = self._fieldict[var]['ndim']
		if idim >= ndim: raiseError(f'Requested dimension {idim} for {var} greater than its number of dimensions {ndim}!')
		print(len(self))
		return  np.ascontiguousarray(self._fieldict[var]['value'][idim:ndim*len(self):ndim])

	def info(self,var):
		'''
		Returns the information for a certain variable
		'''
		return {'point':self._point,'ndim':self._fieldict[var]['ndim']}

	def add_field(self,varname,ndim,var):
		'''
		Add a field to the dataset
		'''
		self._fieldict[varname] = {
			'ndim'  : ndim,
			'value' : var, 
		}

	def add_variable(self,varname,idim,var):
		'''
		Add a variable to the dataset
		'''
		self._vardict[varname] = {
			'idim'  : idim,
			'value' : var, 
		}

	def append_variable(self,varname,var,**fieldict):
		'''
		Appends new timesteps to the dataset
		'''
		# Add to variable vector
		self.vars[varname]['value'] = np.concatenate((self.vars[varname]['value'],var))
		# Sort ascendingly and retrieve sorting index
		idx = np.argsort(self.vars[varname]['value'])
		self.vars[varname]['value'] = self.vars[varname]['value'][idx]
		idim = self.vars[varname]['idim']
		# Now concatenate and sort per variable
		for v in fieldict:
			aux = np.concatenate((self[v][:,:,idim],fieldict[v]),axis=1)[:,idx]
			self[v][:,:,idim] = aux

	def select_random_sensors(self, nsensors, bounds, seed=-1):
		'''
		Generates a set of coordinats of nsensors random sensors inside the region defined by bounds.
		Then for each sensor finds the nearest point from the dataset to get its coordinates and dataset value.
		It creates a new dataset containing all the sensor coordinates and values
		'''
		np.random.seed(0) if seed == -1 else np.random.seed(seed)

		# Generate random points using numpy's uniform distribution
		x = np.random.uniform(bounds[0], bounds[1], nsensors)
		y = np.random.uniform(bounds[2], bounds[3], nsensors)
		z = np.random.uniform(bounds[4], bounds[5], nsensors) if len(bounds) > 4 else None
		# Stack them into an Nxndim
		randcoords = np.vstack((x, y, z)).T if z is not None else np.vstack((x, y)).T 
		 
		senscoord = np.zeros((nsensors, self.xyz.shape[1]))                # Sensor real coordinates
		sensdata  = np.zeros((nsensors, self['VELOX'].shape[1])) # Sensor data
		for ii, sensor in enumerate(randcoords):
			dist = np.sum((sensor-self.xyz)**2, axis=1)
			imin = np.argmin(dist)
			senscoord[ii,:] = self.xyz[imin]
			sensdata[ii,:]  = self['VELOX'][imin]
		ptable = PartitionTable.new(1, 0, nsensors)
		return self.__class__(xyz=senscoord, point=True, ptable=ptable, vars ={'time':{'idim':0,'value':self.get_variable('time')}},VELOX = {'ndim':1,'value':sensdata})

	@cr('Dataset.reshape')
	def reshape(self,field,info):
		'''
		Reshape a field for a single variable
		according to the info
		'''
		# Obtain number of points from the mesh
		npoints = len(self)
		# Only reshape the variable if ndim > 1
		return np.ascontiguousarray(field.reshape((npoints,info['ndim']),order='C') if info['ndim'] > 1 else field)

	@cr('Dataset.X')
	def X(self,*args):
		'''
		Return the X matrix for the selected fields
		'''
		# Select all variables if none is provided
		fieldnames = self.fieldnames if len(args) == 0 else args
		# Compute the number of fields
		npoints = len(self)
		nfields = 0
		for f in fieldnames:
			nfields += self.fields[f]['ndim']
		dims = [nfields*npoints]
		# Variable order could be random, thus create a list of variable
		# names and their idim to order
		varls = np.array(list(self.varnames))
		ivars = np.array([self.vars[v]['idim'] for v in varls])
		idx   = np.argsort(ivars)
		# Order the variables
		varls = varls[idx]
		ivars = ivars[idx]
		# Loop the number of variables according to their idim
		# As minimum we will have 1 variable, thus idim=0. If 
		# we have idim > 0, this surely indicates a multi-dimensional
		# field
		varc = 0
		for v in varls:
			ivar = self.vars[v]['idim']
			lvar = len(self.vars[v]['value'])
			if ivar == varc:
				dims += [lvar]
				varc += 1
		# Create output array
		X = np.zeros(dims,np.double)
		# Populate output matrix
		ifield = 0
		for field in fieldnames:
			v = self.fields[field]
			for idim in range(v['ndim']):
				X[ifield:nfields*npoints:nfields] = v['value'][idim:v['ndim']*npoints:v['ndim']]
				ifield += 1
		return X

	@cr('Dataset.save')
	def save(self,fname,**kwargs):
		'''
		Store the field in various formats.
		'''
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		# Pickle format
		if fmt.lower() == 'pkl': 
			io.pkl_save(fname,self)
		# H5 format
		if fmt.lower() == 'h5':
			# Set default parameters
			if not 'mode' in kwargs.keys():        kwargs['mode']        = 'w' if not os.path.exists(fname) else 'a'
			if not 'mpio' in kwargs.keys():        kwargs['mpio']        = True
			if not 'nopartition' in kwargs.keys(): kwargs['nopartition'] = False
			# Append or save
			if not kwargs.pop('append',False):
				io.h5_save_dset(fname,self.xyz,self.vars,self.fields,self.ordering,self.point,self.partition_table,**kwargs)
			else:
				io.h5_append_dset(fname,self.xyz,self.vars,self.fields,self.ordering,self.point,self.partition_table,**kwargs)

	@classmethod
	@cr('Dataset.load')
	def load(cls,fname,**kwargs):
		'''
		Load a field from various formats
		'''
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		# Pickle format
		if fmt.lower() == 'pkl': 
			return io.pkl_load(fname)
		# H5 format
		if fmt.lower() == 'h5':
			if not 'mpio' in kwargs.keys(): kwargs['mpio'] = True
			xyz, order, point, ptable, varDict, fieldDict = io.h5_load_dset(fname,**kwargs)
			return cls(xyz,ptable,varDict,order, point, **fieldDict)
		raiseError('Cannot load file <%s>!'%fname)

	# Properties
	@property
	def xyz(self):
		return self._xyz
	def x(self):
		return self._xyz[:,0]
	@property
	def y(self):
		return self._xyz[:,1]
	@property
	def z(self):
		return self._xyz[:,2]

	@property
	def ordering(self):
		return self._order
	@property
	def point(self):
		return self._point
	@property
	def partition_table(self):
		return self._ptable

	@property
	def vars(self):
		return self._vardict
	@property
	def varnames(self):
		return list(self._vardict.keys())

	@property
	def fields(self):
		return self._fieldict
	@property
	def fieldnames(self):
		return list(self._fieldict.keys())