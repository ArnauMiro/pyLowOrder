#!/usr/bin/env python
#
# pyLOM, dataset.
#
# Dataset class, reader and reduction routines.
#
# Last rev: 30/07/2021
from __future__ import print_function, division

import os, numpy as np

from .partition_table import PartitionTable
from .                import inp_out as io
from .utils.mpi       import MPI_RANK, MPI_SIZE, mpi_gather
from .utils           import cr_nvtx as cr, raiseError, raiseWarning, gpu_to_cpu, cpu_to_gpu, pprint
from .vmmath          import data_splitting, find_random_sensors


class Dataset(object):
	'''
	The Dataset class wraps the position of the nodes and the time instants
	with the number of variables and relates them so that the operations 
	in parallel are easier.

	Attributes:
		_xyz (np.ndarray) :    coordinates of the points.
		_vardict (dict) : dictionary containing the variable name and their value.
		_fieldict (dict) : dictionary containing the field name and their value.
		_ptable (PartitionTable) : partition table used.
		_order (np.ndarray) :  ordering of the points.
		_point (bool) :  ``True`` if point data, ``False`` if cell data.
	'''
	def __init__(self, xyz=None, ptable=None, vars=None, order=None, point=True, **kwargs):
		'''
		Class constructor
		
		Arguments:
			xyz (np.ndarray, optional) : coordinates of the points.
			ptable (PartitionTable, optional) : partition table used.
			vars (dict, optional) : dictionary containing the variable name and their value. 
			order (np.ndarray) :  ordering of the points (automatically created if none).
			point (bool, optional) :  ``True`` (as default) if point data, ``False`` if cell data.
			kwargs:  dictionary containing the field name and their value.
		'''
		self._xyz      = xyz
		self._vardict  = vars
		self._fieldict = kwargs
		self._ptable   = ptable
		self._order    = np.arange(xyz.shape[0]) if order is None else order
		self._point    = point

	def __len__(self):
		'''
		Returns:
			int : Number of points.
		'''
		return self._xyz.shape[0]

	def __str__(self):
		'''
		String representation.

		Returns:
			str : Summary of the dataset state.
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
			fstr   = nanstr + '- max = ' + str(np.nanmax(field)) \
										 + ', min = ' + str(np.nanmin(field)) \
										 + ', avg = ' + str(np.nanmean(field)) \
					if len(field) > 0 else '- empty!'
			s     += '  > ' +  key + fstr + '\n'
		return s
		
	# Set and get functions
	def __getitem__(self,key):
		'''
		Dataset[key]

		Recover the value of a field given its key

		Args:
			key (str) : key of the field

		Returns:
			(np.ndarray) : value of the field
		'''
		return self._fieldict[key]['value']

	def __setitem__(self,key,value):
		'''
		Dataset[key] = value

		Set the field of a variable given its key

		Args:
			key (str) : key of the field
			value (np.ndarray) : value to be set the key
		'''
		self._fieldict[key]['value'] = value

	# Functions
	def rename(self,new,old):
		'''
		Rename a variable inside a field.

		Args:
			new (str) : new key of the field.
			old (str) : old key of the field.

		Returns:
			Dataset: self.
		'''
		self.fields[new] = self.fields.pop(old)
		return self

	def delete(self,varname):
		'''
		Delete a variable inside a field.

		Args:
			varname (str) : key of the field to remove.

		Returns:
			(np.ndarray) : value of the removed variable.
		'''
		return self.fields.pop(varname)

	def get_variable(self,key:str):
		r'''
		Recover the value of a variable given its key

		Args:
			key (str): name of the variable

		Returns:
			(np.ndarray): value of the variable
		'''
		return self._vardict[key]['value']

	def set_variable(self,key,value):
		'''
		Set the value of a variable given its key

		Args:
			key (str): name of the variable
			value (np.ndarray) : value of the variable
		'''
		self._vardict[key]['value'] = value

	def get_dim(self,var,idim):
		'''
		Recover the value of a variable for a given dimension
		Aborts if ``idim`` is invalid.

		Args:
			var (str) : name of the variable
			idim (int) : requested dimension

		Returns:
			np.ndarray: requested dimension of the variable
		'''
		ndim = self._fieldict[var]['ndim']
		if idim >= ndim: raiseError(f'Requested dimension {idim} for {var} greater than its number of dimensions {ndim}!')
		print(len(self))
		return  np.ascontiguousarray(self._fieldict[var]['value'][idim:ndim*len(self):ndim])

	def info(self,var):
		'''
		Returns the information for a certain variable

		Args:
			var (str) : name of the variable

		Returns:
			dict: contains point and ndim data.
		'''
		return {'point':self._point,'ndim':self._fieldict[var]['ndim']}
	
	def to_gpu(self,fields=None):
		'''
		Send field data to the GPU

		Args:
			fields (list[str], optional) : list with names of the variables

		Returns:
			Dataset : self
		'''
		fields = fields if not fields is None else self.fieldnames
		for key in fields:
			self._fieldict[key]['value'] = cpu_to_gpu(self._fieldict[key]['value'])
		return self

	def to_cpu(self,fields=None):
		'''
		Send field data to the CPU

		Args:
			fields (list[str], optional) : list with names of the variables

		Returns:
			Dataset : self
		'''
		fields = fields if not fields is None else self.fieldnames
		for key in fields:
			self._fieldict[key]['value'] = gpu_to_cpu(self._fieldict[key]['value'])
		return self

	def add_field(self,varname,ndim,var):
		'''
		Add a field to the dataset

		Args:
			varname (str): name of the field
			ndim (int) : number of dimensions of the field
			var (np.ndarray) : value of the field
		'''
		self._fieldict[varname] = {
			'ndim'  : ndim,
			'value' : var, 
		}

	def add_variable(self,varname,idim,var):
		'''
		Add a variable to the dataset

		Args:
			varname (str): name of the variable
			ndim (int) : number of dimensions of the variable
			var (np.ndarray) : value of the variable
		'''
		self._vardict[varname] = {
			'idim'  : idim,
			'value' : var, 
		}

	def split_data(self,var,mode='reconstruct'):
		r'''
		Generate random training, validation and test masks for a dataset of Nt samples.

		Args:
			variable (str): variable which will be splitted in different samples
			mode (str, optional): type of splitting to perform (default, ``'reconstruct'``). In reconstruct mode all three datasets have samples along all the data range.
	
		Returns:
			[(np.ndarray), (np.ndarray), (np.ndarray)]: List of arrays containing the identifiers of the training, validation and test samples.
		'''
		
		N    = len(self.vars[var]["value"])
		idim = self.vars[var]["idim"]
		trid, vaid, teid = data_splitting(N, mode)
		self.add_variable('training_%s'%var,idim,trid)
		self.add_variable('validation_%s'%var,idim,vaid)
		self.add_variable('test_%s'%var,idim,teid)

		return trid, vaid, teid

	def mask_field(self, key, mask):
		'''
		Mask a field over a defined variable

		Args:
			key (str) : name of the variable
			mask : mask to apply

		Returns:	
			np.ndarray : the masked array
		'''
		mask = mask if mask is not str else self.get_variable(mask)
		return self[key][:,mask].copy()

	def append_variable(self,varname,var,**fieldict):
		'''
		Appends new timesteps to the dataset

		Args:
			varname (str) : name of the variable
			var (np.ndarray) : timesteps to append
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

	def select_random_sensors(self, nsensors, bounds, VARLIST, seed=-1):
		'''
		Generates a set of coordinates of ``nsensors`` random sensors inside the
		region defined by ``bounds``.
		Then for each sensor finds the nearest point from the dataset to get its
		coordinates and dataset value.
		It creates a new dataset containing all the sensor coordinates and
		values.

		Args:
			nsensors (int) : number of sensors
			bounds (np.ndarray) : bounds of the region
			VARLIST (list[str]) : list of variables to extract as sensors
			seed (int, optional) : seed the random engine. If negative (as
				default) no seeding is performed

		Returns:
			Dataset : with the data of the selected variables at the random sensors
		'''
		# Fix seed if user requested
		if seed > 0: np.random.seed(seed)

		# Obtain the indices of the sensors and to which rank
		# this index has been found
		idxsensors, ranksensors = find_random_sensors(bounds,self.xyz,nsensors)

		# Create a new partition table
		nparts   = MPI_SIZE
		Nsensors = len(idxsensors)
		points   = mpi_gather(Nsensors, all=True) if MPI_SIZE > 1 else np.array([Nsensors],dtype=np.int32)
		ids      = np.arange(1,nparts+1,dtype=np.int32)
		elements = np.zeros((MPI_SIZE,),dtype=np.int32)
		ptable   = PartitionTable(nparts,ids,elements,points,has_master=False)

		# Find which indices belong to the current rank
		myidx    = idxsensors[ranksensors == MPI_RANK]
		myxyz    = self.xyz[myidx] if len(myidx) > 0 else np.empty((0,self.xyz.shape[1]),self.xyz.dtype)

		# Initialize new dataset
		sp, ep   = ptable.partition_bounds(MPI_RANK)
		order    = np.linspace(start=sp,stop=ep-1,num=ep-sp,dtype=np.int32)
		sd       = self.__class__(xyz=myxyz,ptable=ptable,order=order,point=True,vars=self._vardict)
	
		# Fill in the fields
		for name in self.fieldnames:
			# Skip field that is not in the list
			if name not in VARLIST: continue
			# Skip multi-dimensional fields
			if self.fields[name]["ndim"] > 1:
				raiseWarning("Multidimensional variables are skipped as sensor datasets must be saved in nopartition mode. Separate each dimension of your variable")
				continue
			# Add field
			f = self[name][myidx] if len(myidx) > 0 else np.empty((0,self[name].shape[1]),self[name].dtype)
			sd.add_field(name,1,f)
		return sd

	@cr('Dataset.reshape')
	def reshape(self,field,info):
		'''
		Reshape a field for a single variable according to the info

		Args:
			field (np.ndarray) : field to reshape
			info (dict) : contains ``'ndim'`` the number of dimensions of
				``field``

		Returns:
			np.ndarray: The reshaped field array.
		'''
		# Obtain number of points from the mesh
		npoints = len(self)
		# Only reshape the variable if ndim > 1
		return np.ascontiguousarray(field.reshape((npoints,info['ndim']),order='C') if info['ndim'] > 1 else field)

	@cr('Dataset.X')
	def X(self,*args,dtype=np.double):
		'''
		Return the X matrix for the selected fields

		Args:
			*args : ``None`` or strings of the variables to extract. If ``None``
				all the fields will be returned
			dtype : underlying type of the returned matrix
		Returns:
			X (np.ndarray) : contains all requested fields
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
			ivar   = self.vars[v]['idim']
			lvar   = len(self.vars[v]['value'])
			if ivar == varc:
				dims += [lvar]
				varc += 1
		# Create output array
		X = np.zeros(dims,dtype)
		# Populate output matrix
		ifield = 0
		for field in fieldnames:
			v = self.fields[field]
			for idim in range(v['ndim']):
				X[ifield:nfields*npoints:nfields] = v['value'][idim:v['ndim']*npoints:v['ndim']].astype(dtype)
				ifield += 1
		return X

	@cr('Dataset.save')
	def save(self,fname,**kwargs):
		'''
		Store the field in various formats.

		Args:
			fname (str) : File name
			**kwargs :
				- 'mode' (str) : ``'w'`` for overwrite (default) and ``'a'`` for
					append.
				- 'mpio' (bool) : ``True`` (default) for parallel and ``False``
					for serial.
				- 'nopartition' (bool) : ``True`` to not store the partition table
				  and ``False`` (default) to store it.
				- The rest of arguments are forwarded to ``io.h5_save_dset`` or
				  ``io.h5_append_dset``.
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
		Load a field from various formats.

		Args:
			fname (str) : File name
			**kwargs :
				- 'mpio' (bool) : ``True`` (default) for parallel and ``False``
					for serial.
				- The rest of arguments are forwarded to ``io.h5_load_dset``.
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
