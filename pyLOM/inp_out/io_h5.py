#!/usr/bin/env python
#
# pyLOM, IO
#
# H5 Input Output
#
# Last rev: 31/07/2021
from __future__ import print_function, division

import os, numpy as np, h5py

from ..partition_table import PartitionTable
from ..mesh            import MTYPE2ID, ID2MTYPE
from ..utils           import cr, MPI_COMM, MPI_RANK, MPI_SIZE, worksplit, writesplit, is_rank_or_serial, mpi_reduce, mpi_gather, raiseError


PYLOM_H5_VERSION = (3,0)


def h5_save_partition(file,ptable):
	'''
	Save a partition table inside an HDF5 file
	'''
	# Create a group for the mesh
	group = file.create_group('PARTITIONS')
	group.create_dataset('NSubD',(1,),dtype='i4',data=ptable.n_partitions)
	group.create_dataset('Ids',(ptable.n_partitions,),dtype='i4',data=ptable.Ids)
	group.create_dataset('Elements',(ptable.n_partitions,),dtype='i4',data=ptable.Elements)
	group.create_dataset('Points',(ptable.n_partitions,),dtype='i4',data=ptable.Points)

def h5_load_partition(file):
	'''
	Load a partition table inside an HDF5 file
	'''
	# Load file
	if not 'PARTITIONS' in file.keys(): raiseError('No partition table stored in dataset!')
	nparts   = int(file['PARTITIONS']['NSubD'][0])
	ids      = np.array(file['PARTITIONS']['Ids'][:])
	elements = np.array(file['PARTITIONS']['Elements'][:])
	points   = np.array(file['PARTITIONS']['Points'][:])
	# Return partition class
	return PartitionTable(nparts,ids,elements,points)

def h5_save_meshes(file,mtype,xyz,conec,eltype,cellO,pointO,ptable):
	'''
	Save the mesh inside the HDF5 file
	'''
	# Save the mesh type
	file.create_dataset('type',(1,),dtype='i4',data=MTYPE2ID[mtype])
	# Write the total number of cells and the total number of points
	# Assume we might be dealing with a parallel mesh
	ndim     = xyz.shape[1]
	nnodcell = conec.shape[1]
	npointG  = mpi_reduce(xyz.shape[0],op='sum',all=True)
	ncellG   = mpi_reduce(eltype.shape[0],op='sum',all=True)
	if ptable.has_master: 
		npointG -= 1
		ncellG  -= 1
	file.create_dataset('npoints',(1,),dtype='i4',data=npointG)
	file.create_dataset('ncells' ,(1,),dtype='i4',data=ncellG)
	# Create the rest of the datasets for parallel storage
	dxyz   = file.create_dataset('xyz',(npointG,ndim),dtype=xyz.dtype)
	dconec = file.create_dataset('connectivity',(ncellG,nnodcell),dtype='i4')
	deltyp = file.create_dataset('eltype',(ncellG,),dtype='u1')
	dcellO = file.create_dataset('cellOrder',(ncellG,),dtype='i4')
	dpoinO = file.create_dataset('pointOrder',(npointG,),dtype='i4')
	# Skip master if needed
	if ptable.has_master and MPI_RANK == 0: return None, None, None
	# Point dataset
	# Compute start and end of read, node data
	istartp, iend = ptable.partition_bounds(MPI_RANK,points=True)
	dxyz[istartp:iend,:]  = xyz
	dpoinO[istartp:iend]  = pointO
	# Compute start and end of read, cell data
	istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
	dconec[istart:iend,:] = conec + istartp
	deltyp[istart:iend]   = eltype
	dcellO[istart:iend]   = cellO

def h5_save_meshes_nopartition(file,mtype,xyz,conec,eltype,cellO,pointO,ptable):
	'''
	Save the mesh inside the HDF5 file
	'''
	# Save the mesh type
	file.create_dataset('type',(1,),dtype='i4',data=MTYPE2ID[mtype])
	# Write the total number of cells and the total number of points
	# Assume we might be dealing with a parallel mesh
	ndim     = xyz.shape[1]
	nnodcell = conec.shape[1]
	npointG  = mpi_reduce(pointO.max() if pointO.shape[0] > 0 else 0,op='max',all=True) + 1
	ncellG   = mpi_reduce(cellO.max() if cellO.shape[0] > 0 else 0,op='max',all=True)  + 1
	file.create_dataset('npoints',(1,),dtype='i4',data=npointG)
	file.create_dataset('ncells' ,(1,),dtype='i4',data=ncellG)
	# Create the rest of the datasets for parallel storage
	dxyz   = file.create_dataset('xyz',(npointG,ndim),dtype=xyz.dtype)
	dconec = file.create_dataset('connectivity',(ncellG,nnodcell),dtype='i4')
	deltyp = file.create_dataset('eltype',(ncellG,),dtype='u1')
	dcellO = file.create_dataset('cellOrder',(ncellG,),dtype='i4')
	dpoinO = file.create_dataset('pointOrder',(npointG,),dtype='i4')
	# Skip master if needed
	if ptable.has_master and MPI_RANK == 0: return None, None, None
	# Point dataset
	# Get the position where the points should be stored
	inods,idx = np.unique(pointO,return_index=True)
	dxyz[inods,:] = xyz[idx,:]
	dpoinO[inods] = pointO[idx]
	# Compute start and end of read, cell data
	istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
	dconec[istart:iend,:] = pointO[conec] if pointO.shape[0] > 0 else conec
	deltyp[istart:iend]   = eltype
	dcellO[istart:iend]   = cellO

def h5_load_meshes_size(file):
	'''
	Load only the number of cells and points for the partition
	'''
	# If the mesh is present read the size
	npoints = int(file['npoints'][0])
	ncells  = int(file['ncells'][0])
	return npoints, ncells

def h5_load_meshes(file,ptable,repart):
	'''
	Load the mesh inside the HDF5 file
	'''
	# Read mesh type
	mtype  = ID2MTYPE[int(file['type'][0])]
	# Read cell related variables
	istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
	conec  = np.array(file['connectivity'][istart:iend,:],np.int32)
	eltype = np.array(file['eltype'][istart:iend],np.int32) 
	cellO  = np.array(file['cellOrder'][istart:iend],np.int32)
	cellO  = np.arange(istart, iend, 1) # Fix IO for clipped datasets in pyQvarsi
	# Read point related variables
	if repart:
		# Warning! Repartition will only work if the input file is serial
		# i.e., it does not have any repeated nodes, otherwise it wont work
		ptable.create_partition_points(conec)
		inods  = ptable.partition_points(1)
	else:
		istart, iend = ptable.partition_bounds(MPI_RANK,points=True)
		inods = np.arange(istart,iend,dtype=np.int32)
	xyz    = np.array(file['xyz'][inods,:],file['xyz'].dtype) 
	pointO = np.array(file['pointOrder'][inods],np.int32)
	# Fix the connectivity to start at zero
	conec2 = -np.ones_like(conec).flatten()# This is a 1D array of -1 of the size of our connectivity
	conec2[conec.flatten() >= 0] = np.searchsorted(pointO, conec[conec >= 0].flatten()) # Search only the positive values
	conec = conec2.reshape(conec.shape).astype(np.int32) # Reshape the connectivity to its original format
	print(conec,conec[conec<0])
	# Return
	return mtype, xyz, conec, eltype, cellO, pointO

def h5_save_points(file,xyz,order,ptable,point):
	'''
	Save the points inside the HDF5 file
	'''
	npointG = mpi_reduce(xyz.shape[0] if not np.any(np.isnan(xyz)) else 0,op='sum',all=True)
	ndim    = xyz.shape[1]
	if ptable.has_master: npointG -= 1
	file.create_dataset('pointData',(1,),dtype='i4',data=point)
	file.create_dataset('npoints',(1,),dtype='i4',data=npointG)
	# Create the rest of the datasets for parallel storage
	dxyz   = file.create_dataset('xyz',(npointG,ndim),dtype=xyz.dtype)
	dpoinO = file.create_dataset('order',(npointG,),dtype='i4')
	# Skip master if needed
	if ptable.has_master and MPI_RANK == 0: return None, None, None
	# Compute start and end of read, node data
	istart, iend = ptable.partition_bounds(MPI_RANK,points=point)
	dxyz[istart:iend,:] = xyz
	dpoinO[istart:iend] = order
	return None, None, None

def h5_save_points_nopartition(file,xyz,order,ptable,point):
	'''
	Save the points inside the HDF5 file
	'''
	# Assume we might be dealing with a parallel mesh
	npointG = mpi_reduce(order.max() if order.shape[0] > 0 else 0,op='max',all=True) + 1
	ndim    = xyz.shape[1]
	file.create_dataset('pointData',(1,),dtype='i4',data=point)
	file.create_dataset('npoints',(1,),dtype='i4',data=npointG)
	# Create the rest of the datasets for parallel storage
	dxyz   = file.create_dataset('xyz',(npointG,ndim),dtype=xyz.dtype)
	dpoinO = file.create_dataset('order',(npointG,),dtype='i4')
	# Skip master if needed
	if ptable.has_master and MPI_RANK == 0: return None, None, None
	# Skip empty part
	if order.shape[0] == 0: return None, None, None
	# Get the position where the points should be stored
	inods,idx = np.unique(order,return_index=True)
	# Write dataset - points
	dxyz[inods,:] = xyz[idx,:]
	dpoinO[inods] = order[idx]
	return inods,idx,npointG

def h5_load_dset_size(file):
	'''
	Load only the number of points for the dataset
	'''
	# Crash if mesh is not present
	if not 'xyz' in file.keys():
		raiseError('Repartition is not possible without a the points!')
	# If the mesh is present read the size
	npoints = int(file['npoints'][0])
	point   = int(file['pointData'][0])
	return npoints, point

def h5_load_points(file,ptable,point):
	'''
	Load the mesh inside the HDF5 file
	'''
	if ptable.nodes is None or not point:
		# Warning! Repartition will only work if the input file is serial
		# i.e., it does not have any repeated nodes, otherwise it wont work
		istart, iend = ptable.partition_bounds(MPI_RANK,points=point)
		ptable.nodes = np.arange(istart,iend,dtype=np.int32)
	inods = ptable.nodes
	xyz   = np.array(file['xyz'][inods,:]) 
	order = np.array(file['order'][inods])
	# Return
	return xyz, order

def h5_create_variable_datasets(file,varDict,ptable,ipart=-1):
	'''
	Create the variable datasets inside an HDF5 file
	'''
	# Create group for variables
	group = file.create_group('VARIABLES_%d'%ipart if ipart >= 0 else 'VARIABLES')
	dsetDict = {}
	for var in varDict.keys():
		vargroup = group.create_group(var)
		nvars    = varDict[var]['value'].shape[0]
		dsetDict[var] = {
			'idim'  : vargroup.create_dataset('idim' ,(1,),dtype='i4'),
			'value' : vargroup.create_dataset('value',(nvars,),dtype=varDict[var]['value'].dtype),
		}
	return dsetDict

def h5_fill_variable_datasets(dsetDict,varDict):
	'''
	Fill in the variable datasets inside an HDF5 file
	'''
	for var in dsetDict.keys():
		# Fill dataset
		dsetDict[var]['idim'][:]  = varDict[var]['idim']
		dsetDict[var]['value'][:] = varDict[var]['value']

def h5_load_variables_single(file):
	'''
	Load the variables inside the HDF5 file
	'''
	varDict = {}
	for v in file['VARIABLES'].keys():
		vargroup = file['VARIABLES'][v]
		varDict[v] = {
			'idim'  : int(vargroup['idim'][0]),
			'value' : np.array(vargroup['value']),
		}
	# Return
	return varDict

def h5_load_variables_multi(file,npart):
	'''
	Load the variables inside the HDF5 file
	'''
	# Scan for variables in first partition and build variable dictionary
	varDict = {}
	for v in file['VARIABLES_0'].keys():
		vargroup = file['VARIABLES_0'][v]
		# Load point and dimensions
		idim = int(vargroup['idim'][0])
		# Now allocate output array
		value =  np.array(vargroup['value'])
		# Generate dictionary
		varDict[v] = {'idim':idim,'value':value}
	# Read variables per partition
	for ipart in range(1,npart):
		# Compute start and end of my partition in time
		vargroup = file['VARIABLES_%d'%ipart][v]
		value    =  np.array(vargroup['value'])
		varDict[v]['value'] = np.concatenate((varDict[v]['value'],value))
	# Return
	return varDict

def h5_create_field_datasets(file,fieldDict,ptable,ipart=-1):
	'''
	Create the variable datasets inside an HDF5 file
	'''
	# Create group for variables
	group = file.create_group('FIELDS_%d'%ipart if ipart >= 0 else 'FIELDS')
	dsetDict = {}
	for var in fieldDict.keys():
		vargroup = group.create_group(var)
		n     = mpi_reduce(fieldDict[var]['value'].shape[0] if not np.any(np.isnan(fieldDict[var]['value'])) else 0,op='sum',all=True)
		if ptable.has_master: n -= 1
		npoin = int(file['xyz'].shape[0])
		ndim  = n//npoin
		dims  = tuple([ndim*npoin] + [fieldDict[var]['value'].shape[ivar+1] for ivar in range(len(fieldDict[var]['value'].shape) - 1)])
		dsetDict[var] = {
			'ndim'  : vargroup.create_dataset('ndim' ,(1,),dtype='i4'),
			'nvar'  : vargroup.create_dataset('nvar' ,(1,),dtype='i4'),
			'vars'  : vargroup.create_dataset('vars' ,(len(fieldDict[var]['value'].shape) - 1,),dtype='i4'),
			'value' : vargroup.create_dataset('value',dims,dtype=fieldDict[var]['value'].dtype),
		}
	return dsetDict

def h5_fill_field_datasets(dsetDict,fieldDict,ptable,point,inods,idx):
	'''
	Fill in the variable datasets inside an HDF5 file
	'''
	# Skip master if needed
	if ptable.has_master and MPI_RANK == 0: return
	for var in dsetDict.keys():
		# Fill dataset
		dsetDict[var]['ndim'][:]  = fieldDict[var]['ndim']
		dsetDict[var]['nvar'][:]  = len(fieldDict[var]['value'].shape) - 1
		dsetDict[var]['vars'][:]  = fieldDict[var]['value'].shape[1:]
		# Compute start and end bounds for the variable
		if inods is None:
			istart, iend = ptable.partition_bounds(MPI_RANK,ndim=fieldDict[var]['ndim'],points=point)
			dsetDict[var]['value'][istart:iend,:] = fieldDict[var]['value']
		else:
			if fieldDict[var]['ndim'] > 1: raiseError('Cannot deal with multi-dimensional arrays in no partition mode!')
			dsetDict[var]['value'][inods,:] = fieldDict[var]['value'][idx,:]

def h5_load_fields_single(file,npoints,ptable,varDict,point):
	'''
	Load the fields inside the HDF5 file
	'''
	# Read variables
	fieldDict = {}
	for v in file['FIELDS'].keys():
		fieldgroup = file['FIELDS'][v]
		# Load point and dimensions
		ndim = int(fieldgroup['ndim'][0])
		dims = [ndim*npoints] + list(fieldgroup['vars'])
		# Now allocate output array
		value = np.zeros(dims,fieldgroup['value'])
		# Select which points to load
		if point:
			inods = ptable.partition_points(npoints,ndim=ndim)
		else:
			# Use the partition bounds to recover the array
			istart, iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=False)
			inods = np.arange(istart,iend,dtype=np.int32)
		# Read the values
		value[:] = np.array(fieldgroup['value'][inods])
		# Generate dictionary
		fieldDict[v] = {'ndim':ndim,'value':value}
	# Return
	return fieldDict

def h5_load_fields_multi(file,npoints,ptable,varDict,point,npart):
	'''
	Load the fields inside the HDF5 file
	'''
	# Scan for variables in first partition and build variable dictionary
	fieldDict = {}
	for v in file['FIELDS_0'].keys():
		fieldgroup = file['FIELDS_0'][v]
		# Load point and dimensions
		ndim = int(fieldgroup['ndim'][0])
		dims = [ndim*npoints] + list(np.sum([file['FIELDS_%d'%ipart][v]['vars'] for ipart in range(npart)],axis=0))
		# Now allocate output array
		value = np.zeros(dims,fieldgroup['value'].dtype)	
		# Generate dictionary
		fieldDict[v] = {'ndim':ndim,'value':value}
	# Generate the partition size
	psize = [len(varDict[vv]['value'])//npart for vv in varDict.keys()]
	# Read variables per partition
	for ipart in range(npart):
		# Compute start and end of my partition in time
		pname  = 'FIELDS_%d'%ipart
		pstart = [ipart*p for p in psize]
		pend   = [(ipart+1)*p for p in psize]
		# Read the partition
		for v in file[pname].keys():
			fieldgroup = file[pname][v]
			# Load ndim
			ndim   = int(fieldgroup['ndim'][0])
			sliced = tuple([np.s_[:]] + [np.s_[i:j] for (i,j) in zip(pstart,pend)])
			# Select which points to load
			if point:
				inods = ptable.partition_points(npoints,ndim=ndim)
			else:
				# Use the partition bounds to recover the array
				istart, iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=False)
				inods = np.arange(istart,iend,dtype=np.int32)
			# Read the values
			fieldDict[v]['value'][sliced] = np.array(fieldgroup['value'][inods])
	# Return
	return fieldDict


@cr('h5IO.save_dset')
def h5_save_dset(fname,xyz,varDict,fieldDict,ordering,point,ptable,mode='w',mpio=True,nopartition=False):
	'''
	Save a Dataset in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		h5_save_dset_mpio(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable,nopartition)
	else:
		h5_save_dset_serial(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable)

def h5_save_dset_serial(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable):
	'''
	Save a Dataset in HDF5 in serial mode
	'''
	# Open file for writing
	file = h5py.File(fname,mode)
	file.attrs['Version'] = PYLOM_H5_VERSION
	# Create dataset group
	group = file.create_group('DATASET')
	# Save points
	inods,idx,_ = h5_save_points(group,xyz,ordering,ptable,point)
	# Store the variables
	h5_fill_variable_datasets(h5_create_variable_datasets(group,varDict,ptable),varDict)
	# Store the fields
	h5_fill_field_datasets(h5_create_field_datasets(group,fieldDict,ptable),fieldDict,ptable,point,inods,idx)
	file.close()

def h5_save_dset_mpio(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable,nopartition):
	'''
	Save a Dataset in HDF5 in parallel mode
	'''
	# Open file
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM)
	file.attrs['Version'] = PYLOM_H5_VERSION
	# Create dataset group
	group = file.create_group('DATASET')
	# Save points
	inods,idx,_ = h5_save_points(group,xyz,ordering,ptable,point) if not nopartition else h5_save_points_nopartition(group,xyz,ordering,ptable,point)
	# Store the variables
	h5_fill_variable_datasets(h5_create_variable_datasets(group,varDict,ptable),varDict)
	# Store the fields
	h5_fill_field_datasets(h5_create_field_datasets(group,fieldDict,ptable),fieldDict,ptable,point,inods,idx)
	file.close()


@cr('h5IO.append_dset')
def h5_append_dset(fname,xyz,varDict,fieldDict,ordering,point,ptable,mode='a',mpio=True,nopartition=False):
	'''
	Save a Dataset in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		h5_append_dset_mpio(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable,nopartition)
	else:
		h5_append_dset_serial(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable)

def h5_append_dset_serial(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable):
	'''
	Save a dataset in HDF5 in serial mode
	'''
	file = h5py.File(fname,mode)
	if not hasattr(h5_append_dset_serial,'ipart'):
		# Input file does not exist, we create it with the whole structure
		file.attrs['Version'] = PYLOM_H5_VERSION
		# Create dataset group
		group = file.create_group('DATASET')
		# Save points
		inods,idx,npoints = h5_save_points(group,xyz,ordering,ptable,point)
		# Start the partition counter
		h5_append_dset_serial.ipart   = 0
		h5_append_dset_serial.inods   = inods
		h5_append_dset_serial.idx     = idx
		h5_append_dset_serial.npoints = npoints
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Obtain from function
	group   = file['DATASET']
	ipart   = h5_append_dset_serial.ipart
	inods   = h5_append_dset_serial.inods
	idx     = h5_append_dset_serial.idx
	npoints = h5_append_dset_serial.npoints 
	# Store the variables
	h5_fill_variable_datasets(h5_create_variable_datasets(group,varDict,ptable,ipart=ipart),varDict)
	# Store the fields
	h5_fill_field_datasets(h5_create_field_datasets(group,fieldDict,ptable,ipart=ipart),fieldDict,ptable,point,inods,idx)
	# Increase the partition counter
	h5_append_dset_serial.ipart += 1
	file.close()

def h5_append_dset_mpio(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable,nopartition):
	'''
	Save a dataset in HDF5 in parallel mode
	'''
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM)
	if not hasattr(h5_append_dset_mpio,'ipart'):
		# Input file does not exist, we create it with the whole structure
		file.attrs['Version'] = PYLOM_H5_VERSION
		# Create dataset group
		group = file.create_group('DATASET')
		# Save points
		inods,idx,npoints = h5_save_points(group,xyz,ordering,ptable,point) if not nopartition else h5_save_points_nopartition(group,xyz,ordering,ptable,point)
		# Start the partition counter
		h5_append_dset_mpio.ipart   = 0
		h5_append_dset_mpio.inods   = inods
		h5_append_dset_mpio.idx     = idx
		h5_append_dset_mpio.npoints = npoints
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Obtain from function
	group   = file['DATASET']
	ipart   = h5_append_dset_mpio.ipart
	inods   = h5_append_dset_mpio.inods
	idx     = h5_append_dset_mpio.idx
	npoints = h5_append_dset_mpio.npoints 
	# Store the variables
	h5_fill_variable_datasets(h5_create_variable_datasets(group,varDict,ptable,ipart=ipart),varDict)
	# Store the fields
	h5_fill_field_datasets(h5_create_field_datasets(group,fieldDict,ptable,ipart=ipart),fieldDict,ptable,point,inods,idx)
	# Increase the partition counter
	h5_append_dset_mpio.ipart += 1
	file.close()


@cr('h5IO.load_dset')
def h5_load_dset(fname,ptable=None,mpio=True):
	'''
	Load a dataset in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		return h5_load_dset_mpio(fname,ptable)
	else:
		return h5_load_dset_serial(fname,ptable)

def h5_load_dset_serial(fname,ptable):
	'''
	Load a dataset in HDF5 in serial
	'''
	# Open file for writing
	file = h5py.File(fname,'r')
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Open the dataset group
	group  = file['DATASET']
	# Read dataset size
	npoints,point = h5_load_dset_size(group)
	# Are we reading for the same number of partitions?
	if ptable is None or not ptable.check_split():
		# Redo the partitions table
		ptable = PartitionTable.new(MPI_SIZE,npoints,npoints)
	# Read the points
	xyz, order = h5_load_points(group,ptable,point)
	# Figure out how many partitions we have
	npart = np.sum(['VAR' in key for key in group.keys()])
	# Read the variables
	varDict   = h5_load_variables_single(group) if npart == 1 else h5_load_variables_multi(group,npart)
	fieldDict = h5_load_fields_single(group,npoints,ptable,varDict,point) if npart == 1 else h5_load_fields_multi(group,npoints,ptable,varDict,point,npart)
	file.close()
	return xyz, order, point, ptable, varDict, fieldDict

def h5_load_dset_mpio(fname,ptable):
	'''
	Load a field in HDF5 in parallel
	'''
	# Open file for reading
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM)
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Open the dataset group
	group  = file['DATASET']
	# Read dataset size
	npoints,point = h5_load_dset_size(group)
	# Are we reading for the same number of partitions?
	if ptable is None or not ptable.check_split():
		# Redo the partitions table
		ptable = PartitionTable.new(MPI_SIZE,npoints,npoints)
	# Read the points
	xyz, order = h5_load_points(group,ptable,point)
	# Figure out how many partitions we have
	npoints = xyz.shape[0]
	npart   = np.sum(['VAR' in key for key in group.keys()])
	# Read the variables
	varDict   = h5_load_variables_single(group) if npart == 1 else h5_load_variables_multi(group,npart)
	fieldDict = h5_load_fields_single(group,npoints,ptable,varDict,point) if npart == 1 else h5_load_fields_multi(group,npoints,ptable,varDict,point,npart)
	file.close()
	return xyz, order, point, ptable, varDict, fieldDict


@cr('h5IO.save_mesh')
def h5_save_mesh(fname,mtype,xyz,conec,eltype,cellO,pointO,ptable,mode='w',mpio=True,nopartition=False):
	'''
	Save a Mesh in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		h5_save_mesh_mpio(fname,mode,mtype,xyz,conec,eltype,cellO,pointO,ptable,nopartition)
	else:
		h5_save_mesh_serial(fname,mode,mtype,xyz,conec,eltype,cellO,pointO,ptable)

def h5_save_mesh_serial(fname,mode,mtype,xyz,conec,eltype,cellO,pointO,ptable):
	'''
	Save a Mesh in HDF5 in serial mode
	'''
	# Open file for writing
	file = h5py.File(fname,mode)
	file.attrs['Version'] = PYLOM_H5_VERSION
	# Store partition table
	h5_save_partition(file,ptable)
	# Create dataset group
	group = file.create_group('MESH')
	# Save mesh
	h5_save_meshes(group,mtype,xyz,conec,eltype,cellO,pointO,ptable)
	file.close()

def h5_save_mesh_mpio(fname,mode,mtype,xyz,conec,eltype,cellO,pointO,ptable,nopartition):
	'''
	Save a dataset in HDF5 in parallel mode
	'''
	# Open file
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM)
	file.attrs['Version'] = PYLOM_H5_VERSION
	# Store partition table
	h5_save_partition(file,ptable)
	# Create dataset group
	group = file.create_group('MESH')
	# Save mesh
	h5_save_meshes(group,mtype,xyz,conec,eltype,cellO,pointO,ptable) if not nopartition else h5_save_meshes_nopartition(group,mtype,xyz,conec,eltype,cellO,pointO,ptable)
	file.close()


@cr('h5IO.load_mesh')
def h5_load_mesh(fname,mpio=True):
	'''
	Load a mesh in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		return h5_load_mesh_mpio(fname)
	else:
		return h5_load_mesh_serial(fname)

def h5_load_mesh_serial(fname):
	'''
	Load a mesh in HDF5 in serial
	'''
	# Open file for writing
	file = h5py.File(fname,'r')
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Read partition table
	ptable = h5_load_partition(file)
	repart = False
	# Are we reading for the same number of partitions?
	group = file['MESH']
	if not ptable.check_split():
		# Read the number of elements and points to compute
		# the new partition table
		npoints, ncells = h5_load_meshes_size(group)
		# Redo the partitions table
		ptable = PartitionTable.new(MPI_SIZE,ncells,npoints)
		repart = True
	# Read the mesh
	if not 'MESH' in file.keys(): raiseError('Mesh not present in dataset!')
	mtype,xyz,conec,eltype,cellO,pointO = h5_load_meshes(group,ptable,repart)
	# Close the file
	file.close()
	return mtype, xyz, conec, eltype, cellO, pointO, ptable

def h5_load_mesh_mpio(fname):
	'''
	Load a mesh in HDF5 in parallel
	'''
	# Open file for reading
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM)
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Read partition table
	ptable = h5_load_partition(file)
	repart = False
	# Are we reading for the same number of partitions?
	group = file['MESH']
	if not ptable.check_split():
		# Read the number of elements and points to compute
		# the new partition table
		npoints, ncells = h5_load_meshes_size(group)
		# Redo the partitions table
		ptable = PartitionTable.new(MPI_SIZE,ncells,npoints)
		repart = True
	# Read the mesh
	if not 'MESH' in file.keys(): raiseError('Mesh not present in dataset!')
	mtype,xyz,conec,eltype,cellO,pointO = h5_load_meshes(group,ptable,repart)
	# Close the file
	file.close()
	return mtype, xyz, conec, eltype, cellO, pointO, ptable


@cr('h5IO.save_POD')
def h5_save_POD(fname,U,S,V,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store POD variables into an HDF5 file.
	Can be appended to another HDF by setting the
	mode to 'a'. Then no partition table will be saved.
	'''
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM) if not MPI_SIZE == 1 else h5py.File(fname,mode)
	# Store attributes and partition table
	if not mode == 'a':
		file.attrs['Version'] = PYLOM_H5_VERSION
		# Store partition table
		h5_save_partition(file,ptable)
	# Now create a POD group
	group = file.create_group('POD')
	# Create the datasets for U, S and V
	group.create_dataset('pointData',(1,),dtype='u1',data=pointData)
	group.create_dataset('n_variables',(1,),dtype='u1',data=nvars)
	Usize = (mpi_reduce(U.shape[0],op='sum',all=True),U.shape[1])
	dsetU = group.create_dataset('U',Usize,dtype=U.dtype)
	dsetS = group.create_dataset('S',S.shape,dtype=S.dtype)
	dsetV = group.create_dataset('V',V.shape,dtype=V.dtype)
	# Store S and U that are repeated across the ranks
	# So it is enough that one rank stores them
	if is_rank_or_serial(0):
		dsetS[:] = S
		dsetV[:] = V
	# Store U in parallel
	istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,points=pointData)
	dsetU[istart:iend,:] = U
	file.close()

@cr('h5IO.load_POD')
def h5_load_POD(fname,vars,nmod,ptable=None):
	'''
	Load POD variables from an HDF5 file.
	'''
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM) if not MPI_SIZE == 1 else h5py.File(fname,'r')
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Read the requested variables S, V
	varList = []
	if 'U' in vars:
		# Check if we need to read the partition table
		if ptable is None: ptable = h5_load_partition(file)
		# Read
		nvars = int(file['POD']['n_variables'][0])
		point = bool(file['POD']['pointData'][0])
		istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,points=point)
		varList.append( np.array(file['POD']['U'][istart:iend,:nmod]) )
	if 'S' in vars: varList.append( np.array(file['POD']['S'][:]) )
	if 'V' in vars: varList.append( np.array(file['POD']['V'][:,:]) )
	# Return
	file.close()
	return varList


@cr('h5IO.save_DMD')
def h5_save_DMD(fname,muReal,muImag,Phi,bJov,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store DMD variables into an HDF5 file.
	Can be appended to another HDF by setting the
	mode to 'a'. Then no partition table will be saved.
	'''
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM) if not MPI_SIZE == 1 else h5py.File(fname,mode)
	# Store attributes and partition table
	if not mode == 'a':
		file.attrs['Version'] = PYLOM_H5_VERSION
		# Store partition table
		h5_save_partition(file,ptable)
	# Now create a POD group
	group = file.create_group('DMD')
	# Create the datasets for U, S and V
	group.create_dataset('pointData',(1,),dtype='u1',data=pointData)
	group.create_dataset('n_variables',(1,),dtype='u1',data=nvars)
	Phisz = (mpi_reduce(Phi.shape[0],op='sum',all=True),Phi.shape[1])
	dsPhi = group.create_dataset('Phi',Phisz,dtype=Phi.dtype)
	dsMu  = group.create_dataset('Mu',(muReal.shape[0],2),dtype=muReal.dtype)
	dsJov = group.create_dataset('bJov',bJov.shape,dtype=bJov.dtype)
	# Store S and U that are repeated across the ranks
	# So it is enough that one rank stores them
	if is_rank_or_serial(0):
		dsMu[:,0] = muReal
		dsMu[:,1] = muImag
		dsJov[:]  = bJov
	# Store U in parallel
	istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,points=pointData)
	dsPhi[istart:iend,:] = Phi
	file.close()

@cr('h5IO.load_DMD')
def h5_load_DMD(fname,vars,nmod,ptable=None):
	'''
	Load DMD variables from an HDF5 file.
	'''
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM) if not MPI_SIZE == 1 else h5py.File(fname,'r')
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Read the requested variables S, V
	varList = []
	if 'Phi' in vars:
		# Check if we need to read the partition table
		if ptable is None: ptable = h5_load_partition(file)
		# Read
		nvars = int(file['DMD']['n_variables'][0])
		point = bool(file['DMD']['pointData'][0])
		istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,point=point)
		varList.append( np.array(file['DMD']['Phi'][istart:iend,:nmod]) )
	if 'mu' in vars: 
		varList.append( np.array(file['DMD']['Mu'][:,0]) ) # Real
		varList.append( np.array(file['DMD']['Mu'][:,1]) ) # Imag
	if 'bJov' in vars: varList.append( np.array(file['DMD']['bJov'][:]) )
	# Return
	file.close()
	return varList

@cr('h5IO.save_VAE')
def h5_save_VAE(fname, kld, mse, val_loss, train_loss_avg, corrcoef, mode='w'):
	'''
	Store VAE results.
	'''
	file = h5py.File('%s.h5'%fname,mode,driver='mpio',comm=MPI_COMM) if not MPI_SIZE == 1 else h5py.File(fname,mode)
	# Now create a VAE group
	group = file.create_group('VAE')
	# Create the datasets for U, S and V
	
	group.create_dataset('kld',(kld.shape[0],),dtype='u1',data=kld)
	group.create_dataset('mse',(mse.shape[0],),dtype='u1',data=mse)
	group.create_dataset('val_loss',(val_loss.shape[0],),dtype='u1',data=val_loss)
	group.create_dataset('train_loss_avg',(train_loss_avg.shape[0],),dtype='u1',data=train_loss_avg)
	group.create_dataset('correlation',(corrcoef.shape[0],),dtype='u1',data=corrcoef)
	file.close()

def h5_save_SPOD(fname,L,P,f,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store SPOD variables into an HDF5 file.
	Can be appended to another HDF by setting the
	mode to 'a'. Then no partition table will be saved.
	'''
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM) if not MPI_SIZE == 1 else h5py.File(fname,mode)
	# Store attributes and partition table
	if not mode == 'a':
		file.attrs['Version'] = PYLOM_H5_VERSION
		# Store partition table
		h5_save_partition(file,ptable)
	# Get number of blocks
	nblocks = L.shape[1]
	# Now create a POD group
	group = file.create_group('SPOD')
	# Create the datasets for U, S and V
	group.create_dataset('pointData',(1,),dtype='u1',data=pointData)
	group.create_dataset('n_variables',(1,),dtype='u1',data=nvars)
	group.create_dataset('n_blocks',(1,),dtype='u1',data=nblocks)
	Psz = (mpi_reduce(P.shape[0],op='sum',all=True),P.shape[1])
	dsP = group.create_dataset('P',Psz,dtype=P.dtype)
	dsL = group.create_dataset('L',L.shape,dtype=L.dtype)
	dsf = group.create_dataset('f',f.shape,dtype=f.dtype)
	# Store L and f that are repeated across the ranks (nblocks,nfreq)
	# So it is enough that one rank stores them
	if is_rank_or_serial(0):
		dsL[:,:] = L
		dsf[:]   = f
	# Store P in parallel (nblocks*nvars*npoints,nfreq)
	istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars*nblocks,points=pointData)
	dsP[istart:iend,:] = P
	file.close()

def h5_load_SPOD(fname,vars,nmod,ptable=None):
	'''
	Load SPOD variables from an HDF5 file.
	'''
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM) if not MPI_SIZE == 1 else h5py.File(fname,'r')
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Read the requested variables S, V
	varList = []
	if 'P' in vars:
		# Check if we need to read the partition table
		if ptable is None: ptable = h5_load_partition(file)
		# Read
		nvars   = int(file['SPOD']['n_variables'][0])
		nblocks = int(file['SPOD']['n_blocks'][0])
		point   = bool(file['SPOD']['pointData'][0])
		istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars*nblocks,point=point)
		varList.append( np.array(file['SPOD']['P'][istart:iend,:nmod]) )
	if 'L' in vars: 
		varList.append( np.array(file['SPOD']['L'][:,:]) )
	if 'f' in vars: 
		varList.append( np.array(file['SPOD']['f'][:]) )
	# Return
	file.close()
	return varList
