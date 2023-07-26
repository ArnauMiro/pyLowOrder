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
from .utils.cr     import cr
from .utils.mem    import mem
from .utils.errors import raiseError
from .utils.parall import mpi_reduce


class Dataset(object):
	'''
	The Dataset class wraps the position of the nodes and the time instants
	with the number of variables and relates them so that the operations 
	in parallel are easier.
	'''
	@mem('Dataset')
	def __init__(self, ptable=None, mesh=None, time=np.array([0.],np.double), **kwargs):
		'''
		Class constructor

		Inputs:
			> ptable: partition table used.
			> mesh:   mesh class if available.
			> time:   time instants as a numpy array.
			> kwags:  dictionary containin the variable name and values as a
					  python dictionary.
		'''
		self._time    = time
		self._vardict = kwargs
		self._mesh    = mesh
		self._ptable  = ptable

	def __len__(self):
		return self._time.shape[0]

	def __str__(self):
		'''
		String representation
		'''
		s  = 'Dataset of %d instants:\n' % len(self)
		s += '  > time - max = ' + str(np.nanmax(self._time,axis=0)) + ', min = ' + str(np.nanmin(self._time,axis=0)) + '\n'
		for key in self.varnames:
			var    = self[key]
			nanstr = ' (has NaNs) ' if np.any(np.isnan(var)) else ' '
			s     += '  > ' +  key + nanstr + '- max = ' + str(np.nanmax(var)) \
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
		return self._vardict[key]['value']

	def __setitem__(self,key,value):
		'''
		Dataset[key] = value

		Set the value of a variable given its key
		'''
		self._vardict[key]['value'] = value

	# Functions
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

	def add_variable(self,varname,point,ndim,var):
		'''
		Add a variable to the dataset
		'''
		self._vardict[varname] = {
			'point' : point,
			'ndim'  : ndim,
			'value' : var, 
		}

	def append_time(self,time,**varDict):
		'''
		Appends new timesteps to the dataset
		'''
		# Add to time vector
		self.time = np.concatenate((self.time,time))
		# Sort ascendingly and retrieve sorting
		# index
		idx = np.argsort(self.time)
		self.time = self.time[idx]
		# Now concatenate and sort per variable
		for v in varDict:
			self[v] = np.concatenate((self[v],varDict[v]),axis=1)[:,idx]

	@cr('Dataset.X')
	def X(self,*args,time_slice=np.s_[:]):
		'''
		Return the X matrix for the selected variables

		To define a slice in numpy use np.s_ so that:
			X[:,:1000] -> X[:,np.s_[:1000]]
		or
			X[:,::5] -> X[:,np.s_[::5]]
		'''
		# Select all variables if none is provided
		variables = self.varnames if len(args) == 0 else args
		# Compute the number of variables
		nvars = 0
		for var in variables:
			nvars += self.var[var]['ndim']
		# Create output array
		npoints = self.mesh.npoints if self.var[variables[0]]['point'] else self.mesh.ncells
		ninst   = self._time[time_slice].shape[0]
		X = np.zeros((nvars*npoints,ninst),np.double)
		# Populate output matrix
		ivar = 0
		for var in variables:
			v = self.var[var]
			for idim in range(v['ndim']):
				X[ivar:nvars*npoints:nvars,:] = v['value'][idim:v['ndim']*npoints:v['ndim'],time_slice]
				ivar += 1
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
			if not 'mpio' in kwargs.keys():        kwargs['mpio']        = True
			if not 'nopartition' in kwargs.keys(): kwargs['nopartition'] = False
			# Append or save
			if not kwargs.pop('append',False):
				io.h5_save(fname,self.time,self.var,self.mesh,self.partition_table,**kwargs)
			else:
				io.h5_append(fname,self.time,self.var,self.mesh,self.partition_table,**kwargs)

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
			ptable, mesh, time, varDict = io.h5_load(fname,**kwargs)
			return cls(ptable,mesh,time,**varDict)
		raiseError('Cannot load file <%s>!'%fname)

	@cr('Dataset.write')
	def write(self,casestr,basedir='./',instants=[0],times=[0.],vars=[],fmt='vtk'):
		'''
		Store the data using various formats.
		This method differs from save in the fact that save is used 
		to recover the field, write only outputs the data.
		'''
		os.makedirs(basedir,exist_ok=True)
		if fmt.lower() in ['vtk']:
			raiseError('VTK format not yet implemented!')
		elif fmt.lower() in ['ensi','ensight']:
			EnsightWriter(self,casestr,basedir,instants,vars)
		elif fmt.lower() in ['vtkh5','vtkhdf']:
			VTKHDF5Writer(self,casestr,basedir,instants,times,vars)
		else:
			raiseError('Format <%s> not implemented!'%fmt)

	# Properties
	@property
	def time(self):
		return self._time
	@time.setter
	def time(self,value):
		self._time = value

	@property
	def partition_table(self):
		return self._ptable
	@property
	def mesh(self):
		return self._mesh

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
		'eltype' : mesh.eltypeENSI
	}
	# Write geometry file
	io.Ensight_writeGeo(geofile,dset.mesh.xyz,dset.mesh.connectivity+1,header) # Python index start at 0
	# Write instantaneous fields
	binfile_fmt = '%s.ensi.%s-%06d'
	# Define Ensight header
	header = {
		'descr'  : 'File created with pyLOM',
		'partID' : 1,
		'partNM' : 'part',
		'eltype' : mesh.eltypeENSI
	}
	# Loop the selected instants
	for var in varnames:
		# Recover variable information
		info  = dset.info(var)
		field = dset[var]
		# Variable has temporal evolution
		header['eltype'] = mesh.eltypeENSI
		if len(field.shape) > 1:
			# Loop requested instants
			for instant in instants:
				filename = os.path.join(basedir,binfile_fmt % (casestr,var,instant+1))
				# Reshape variable for Ensight file
				f = dset.mesh.reshape_var(field[:,instant],info)
				io.Ensight_writeField(filename,f,header)
		else:
			filename = os.path.join(basedir,binfile_fmt % (casestr,var,1))
			# Reshape variable for Ensight file
			f = dset.mesh.reshape_var(field,dset.mesh,info)
			io.Ensight_writeField(filename,f,header)


def VTKHDF5Writer(dset,casestr,basedir,instants,times,varnames):
	'''
	Ensight dataset writer
	'''
	# Loop the instants
	for instant, time in zip(instants,times):
		filename = os.path.join(basedir,'%s-%08d-vtk.hdf'%(casestr,instant))
		# Write the mesh on the file
		io.vtkh5_save_mesh(filename,dset.mesh,dset.partition_table)
		# Write the data on the file
		varDict = {v:dset.mesh.reshape_var(dset[v][:,instant] if len(dset[v].shape) > 1 else dset[v],dset.info(v)) for v in varnames}
		io.vtkh5_save_field(filename,instant,time,varDict,dset.partition_table)
