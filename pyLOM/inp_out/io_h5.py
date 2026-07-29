#!/usr/bin/env python
#
# pyLOM, IO
#
# H5 Input Output
#
# Last rev: 31/07/2021
from __future__ import print_function, division

import hashlib, numpy as np, h5py

from typing import Optional, Mapping, Union
from collections import OrderedDict

from ..partition_table import PartitionTable
from ..mesh            import MTYPE2ID, ID2MTYPE
from ..utils           import cr, MPI_COMM, MPI_RANK, MPI_SIZE, is_rank_or_serial, mpi_reduce, raiseError, raiseWarning


PYLOM_H5_VERSION = (3,0)

H5_APPEND_MODE         = 'appendMode'
H5_APPEND_CURSOR       = 'appendCursor'
H5_APPEND_BLOCK_SIZE   = 'appendBlockSize'
H5_APPEND_NOPARTITION  = 'appendNoPartition'
H5_APPEND_LAYOUT_HASH  = 'appendLayoutHash'
H5_RESIZABLE_APPEND    = 'resizable'
H5_APPEND_CHUNK_BYTES  = 1024**2
H5_APPEND_HASH_ROWS    = 65536


def h5_resizable_append_cursor(file):
	r'''
	Return the committed length of a resizable append dataset.

	Args:
		file (h5py.Group): ``DATASET`` group in the HDF5 file.

	Returns:
		int or None: number of written entries, or ``None`` for a regular HDF5
			dataset.
	'''
	stored_append_type = file.attrs.get(H5_APPEND_MODE,None)
	if isinstance(stored_append_type,bytes): stored_append_type = stored_append_type.decode()
	if not stored_append_type == H5_RESIZABLE_APPEND: return None
	if not H5_APPEND_CURSOR in file.attrs:
		raiseError('Resizable append cursor is missing!')
	cursor = int(file.attrs[H5_APPEND_CURSOR])
	if cursor < 0: raiseError('Invalid resizable append cursor <%d>!'%cursor)
	return cursor


def h5_resizable_append_block_size(varDict,fieldDict):
	r'''
	Validate a resizable append block and return its common length.

	Resizable append currently supports one appended dimension. Every variable
	therefore has ``idim=0`` and every field is a two-dimensional array whose
	second dimension has the same length as all variables.

	Args:
		varDict (dict): dataset variables for the block.
		fieldDict (dict): dataset fields for the block.

	Returns:
		int: number of entries in the append block.
	'''
	if varDict is None or len(varDict) == 0:
		raiseError('Resizable append requires at least one variable!')
	if fieldDict is None or len(fieldDict) == 0:
		raiseError('Resizable append requires at least one field!')
	block_sizes = []
	for var in sorted(varDict.keys()):
		value = np.asarray(varDict[var]['value'])
		if not int(varDict[var]['idim']) == 0:
			raiseError('Resizable append variable <%s> must have idim=0!'%var)
		if not value.ndim == 1:
			raiseError('Resizable append variable <%s> must be one-dimensional!'%var)
		block_sizes.append(value.shape[0])
	for var in sorted(fieldDict.keys()):
		value = np.asarray(fieldDict[var]['value'])
		if int(fieldDict[var]['ndim']) < 1:
			raiseError('Resizable append field <%s> has an invalid ndim!'%var)
		if not value.ndim == 2:
			raiseError('Resizable append field <%s> must have one append dimension!'%var)
		block_sizes.append(value.shape[1])
	if block_sizes[0] < 1:
		raiseError('Resizable append blocks cannot be empty!')
	if not all(size == block_sizes[0] for size in block_sizes):
		raiseError('All variables and fields in a resizable append must have the same size!')
	return block_sizes[0]


def h5_resizable_append_capacity(block_size,append_total_size):
	r'''
	Return the initial or requested physical capacity of an append dataset.

	``append_total_size`` reserves space but does not set a hard maximum. The
	underlying HDF5 arrays always retain an unlimited append dimension.

	Args:
		block_size (int): size of one append block.
		append_total_size (int or None): requested total capacity.

	Returns:
		int: capacity large enough to contain the current block.
	'''
	if append_total_size is None: return block_size
	if isinstance(append_total_size,(bool,np.bool_)) or not isinstance(append_total_size,(int,np.integer)):
		raiseError('append_total_size must be a positive integer!')
	if append_total_size < 1:
		raiseError('append_total_size must be a positive integer!')
	return max(block_size,int(append_total_size))


def h5_resizable_append_layout_hash(xyz,ordering):
	r'''
	Return a stable MPI-wide fingerprint of a partitioned spatial layout.

	The hash includes each rank's coordinate and ordering shapes, dtypes, and raw
	values. Arrays are processed in bounded row chunks to avoid large temporary
	copies when an input is not C-contiguous.

	Args:
		xyz (np.ndarray): local point coordinates.
		ordering (np.ndarray): global ordering of local points.

	Returns:
		np.ndarray: MPI-wide SHA-256 digest with shape ``(32,)``.
	'''
	digest = hashlib.sha256()
	for array in [np.asarray(xyz),np.asarray(ordering)]:
		digest.update(array.dtype.str.encode('ascii'))
		digest.update(np.asarray(array.shape,dtype='<i8').tobytes())
		for istart in range(0,array.shape[0],H5_APPEND_HASH_ROWS):
			chunk = np.ascontiguousarray(array[istart:istart+H5_APPEND_HASH_ROWS])
			digest.update(memoryview(chunk).cast('B'))
	local_hash = np.frombuffer(digest.digest(),dtype=np.uint8).copy()
	if MPI_SIZE > 1:
		digest = hashlib.sha256()
		for rank_hash in MPI_COMM.allgather(local_hash): digest.update(rank_hash.tobytes())
		return np.frombuffer(digest.digest(),dtype=np.uint8).copy()
	return local_hash


def h5_save_partition(file,ptable):
	r'''
	Save a partition table inside an HDF5 file.

	Args:
		file (string): file name
		ptable (PartitionTable): partition table to be saved
	'''
	# Create a group for the mesh
	group = file.create_group('PARTITIONS')
	group.create_dataset('NSubD',(1,),dtype='i4',data=ptable.n_partitions)
	group.create_dataset('Ids',(ptable.n_partitions,),dtype='i4',data=ptable.Ids)
	group.create_dataset('Elements',(ptable.n_partitions,),dtype='i4',data=ptable.Elements)
	group.create_dataset('Points',(ptable.n_partitions,),dtype='i4',data=ptable.Points)

def h5_load_partition(file):
	r'''
	Load a partition table inside an HDF5 file.

	Args:
		file (string): file name to load the partition from

	Returns:
		PartitionTable
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
	r'''
	Save the mesh inside the HDF5 file

	Args:
		file (string): file name to load the partition from
		mtype
		xyz (np.ndarray): coordinates
		conec (np.ndarray): connectivity
		eltype (np.ndarray): type of element
		cellO (np.ndarray): cell order
		pointO (np.ndarray): point order
		ptable (PartitionTable): partition table
	'''
	# Save attributes
	file.attrs['NOPARTITION'] = False # Either nopartition=False or serial
	file.attrs['PARTS']       = ptable.n_partitions
	# Save the mesh type
	file.create_dataset('type',(1,),dtype='i4',data=MTYPE2ID[mtype])
	# Write the total number of cells and the total number of points
	# Assume we might be dealing with a parallel mesh
	ndim     = xyz.shape[1]
	nnodcell = conec.shape[1]
	npointG  = mpi_reduce(xyz.shape[0],op='sum',all=True) if ptable.n_partitions > 1 else xyz.shape[0]
	ncellG   = mpi_reduce(eltype.shape[0],op='sum',all=True) if ptable.n_partitions > 1 else eltype.shape[0]
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
	dconec[istart:iend,:] = conec
	deltyp[istart:iend]   = eltype
	dcellO[istart:iend]   = cellO

def h5_save_meshes_nopartition(file,mtype,xyz,conec,eltype,cellO,pointO,ptable):
	r'''
	Save the mesh inside the HDF5 file removing the repeated points so that we can change the partition while running the POD.

	Args:
		file (string): file name to load the partition from
		mtype
		xyz (np.ndarray): coordinates
		conec (np.ndarray): connectivity
		eltype (np.ndarray): type of element
		cellO (np.ndarray): cell order
		pointO (np.ndarray): point order
		ptable (PartitionTable): partition table
	'''
	# Save attributes
	file.attrs['NOPARTITION'] = True
	file.attrs['PARTS']       = ptable.n_partitions
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
	r'''
	Load only the number of cells and points for the partition

	Args:
		file (string): file where the mesh is stored
	
	Returns
		int, int: number of points and number of cells
	'''
	# If the mesh is present read the size
	npoints = int(file['npoints'][0])
	ncells  = int(file['ncells'][0])
	return npoints, ncells

def h5_load_meshes(file,ptable,repart):
	r'''
	Load the mesh inside the HDF5 file

	Args:
		file (string): file where the mesh is stored
		ptable (PartitionTable): partition table which the mesh will be loaded
		repart (Bool): whether the mesh has to be repartitioned or not

	Returns
		_, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray: mesh type, coordinates, connectivity, element type, cell order and point order
	'''
	# Check how the mesh was stored
	nopartition = file.attrs.get('NOPARTITION',True)
	nparts      = file.attrs.get('PARTS',MPI_RANK)
	if not nopartition and nparts != MPI_SIZE:
			raiseError(f'Loading a mesh saved in nopartition={nopartition} with different parts (orig: {nparts}, actual: {MPI_SIZE})')
	# Read mesh type
	mtype  = ID2MTYPE[int(file['type'][0])]
	# Read cell related variables
	istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
	conec  = np.array(file['connectivity'][istart:iend,:],np.int32)
	eltype = np.array(file['eltype'][istart:iend],np.int32) 
	# Build cell order from zero using the partition table
	# so that we generate a global array
	cellO  = np.arange(istart,iend,1,dtype=np.int32) 
	# Read point related variables
	if repart:
		# Warning! Repartition will only work if the input file is serial
		# i.e., it does not have any repeated nodes, otherwise it wont work
		ptable.create_partition_points(conec)
		inods  = ptable.partition_points(1)
		if not nopartition: raiseWarning(f'Repartition of mesh will only work if the input file is serial!')
	else:
		istart, iend = ptable.partition_bounds(MPI_RANK,points=True)
		inods = np.arange(istart,iend,dtype=np.int32)
	xyz    = np.array(file['xyz'][inods,:],file['xyz'].dtype) 
	pointO = np.array(file['pointOrder'][inods],np.int32)
	# Fix the connectivity to start at zero
	if nopartition == True:
		conec2 = -np.ones_like(conec).flatten()# This is a 1D array of -1 of the size of our connectivity
		conec2[conec.flatten() >= 0] = np.searchsorted(pointO, conec[conec >= 0].flatten()) # Search only the positive values
		conec  = conec2.reshape(conec.shape).astype(np.int32) # Reshape the connectivity to its original format
	# Return
	return mtype, xyz, conec, eltype, cellO, pointO

def h5_save_points(file,xyz,order,ptable,point):
	'''
	Save the points inside the HDF5 file
	'''
	# Save attributes
	file.attrs['NOPARTITION'] = False # Either nopartition=False or serial
	file.attrs['PARTS']       = ptable.n_partitions
	# Obtain number of points
	npointG = mpi_reduce(xyz.shape[0] if not np.any(np.isnan(xyz)) else 0,op='sum',all=True) if ptable.n_partitions > 1 else xyz.shape[0]
	ndim    = xyz.shape[1]
	if ptable.has_master: npointG -= 1
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
	istart, iend = ptable.partition_bounds(MPI_RANK,points=point)
	inods,idx    = np.arange(istart,iend,dtype=np.int32), np.arange(0,xyz.shape[0],dtype=np.int32)
	# Compute start and end of read, node data
	dxyz[istart:iend,:] = xyz
	dpoinO[istart:iend] = order
	return inods, idx, npointG

def h5_save_points_nopartition(file,xyz,order,ptable,point):
	'''
	Save the points inside the HDF5 file
	'''
	# Save attributes
	file.attrs['NOPARTITION'] = True
	file.attrs['PARTS']       = ptable.n_partitions
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
	nopartition = file.attrs.get('NOPARTITION',True)
	parts       = file.attrs.get('PARTS',1)
	if ptable.nodes is None or not point:
		# Warning! Repartition will only work if the input file is serial
		# i.e., it does not have any repeated nodes, otherwise it wont work
		istart, iend = ptable.partition_bounds(MPI_RANK,points=point)
		ptable.nodes = np.arange(istart,iend,dtype=np.int32)
		if not nopartition and parts != ptable.n_partitions: raiseWarning(f'Repartition of dataset will only work if the input file is serial!')
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
	cursor  = h5_resizable_append_cursor(file)
	varDict = {}
	for v in file['VARIABLES'].keys():
		vargroup = file['VARIABLES'][v]
		varDict[v] = {
			'idim'  : int(vargroup['idim'][0]),
			'value' : np.array(vargroup['value'][:cursor] if not cursor is None else vargroup['value']),
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
		n     = mpi_reduce(fieldDict[var]['value'].shape[0] if not np.any(np.isnan(fieldDict[var]['value'])) else 0,op='sum',all=True) if ptable.n_partitions > 1 else fieldDict[var]['value'].shape[0]
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
			if fieldDict[var]['ndim'] > 1: raiseError('Cannot deal with multi-dimensional arrays when inods are provided!')
			dsetDict[var]['value'][inods,:] = fieldDict[var]['value'][idx,:]


def h5_create_resizable_variable_datasets(file,varDict,capacity,block_size):
	r'''
	Create the regular ``VARIABLES`` group with resizable value arrays.

	Args:
		file (h5py.Group): ``DATASET`` group in the HDF5 file.
		varDict (dict): variables in the first append block.
		capacity (int): initial physical length of each value array.
		block_size (int): expected length of every append block.
	'''
	group = file.create_group('VARIABLES')
	for var in sorted(varDict.keys()):
		vargroup  = group.create_group(var)
		value     = np.asarray(varDict[var]['value'])
		itemsize  = max(1,value.dtype.itemsize)
		chunk_len = min(block_size,max(1,H5_APPEND_CHUNK_BYTES//itemsize))
		vargroup.create_dataset('idim',(1,),dtype='i4',data=int(varDict[var]['idim']))
		vargroup.create_dataset('value',(capacity,),maxshape=(None,),chunks=(chunk_len,),dtype=value.dtype)


def h5_create_resizable_field_datasets(file,fieldDict,capacity,block_size):
	r'''
	Create the regular ``FIELDS`` group with resizable value arrays.

	Only the final, appended dimension is unlimited. The spatial dimension is
	fixed from the first block, and ``vars`` records the logical written length
	rather than the possibly larger reserved capacity.

	Args:
		file (h5py.Group): ``DATASET`` group in the HDF5 file.
		fieldDict (dict): fields in the first append block.
		capacity (int): initial physical length of each value array.
		block_size (int): expected length of every append block.
	'''
	group   = file.create_group('FIELDS')
	npoints = int(file['npoints'][0])
	for var in sorted(fieldDict.keys()):
		vargroup  = group.create_group(var)
		value     = np.asarray(fieldDict[var]['value'])
		ndim      = int(fieldDict[var]['ndim'])
		nrows     = ndim*npoints
		itemsize  = max(1,value.dtype.itemsize)
		timechunk = min(block_size,max(1,H5_APPEND_CHUNK_BYTES//itemsize))
		rowchunk  = min(nrows,max(1,H5_APPEND_CHUNK_BYTES//(itemsize*timechunk)))
		vargroup.create_dataset('ndim',(1,),dtype='i4',data=ndim)
		vargroup.create_dataset('nvar',(1,),dtype='i4',data=1)
		vargroup.create_dataset('vars',(1,),dtype='i4',data=0)
		vargroup.create_dataset('value',(nrows,capacity),maxshape=(nrows,None),chunks=(rowchunk,timechunk),dtype=value.dtype)


def h5_validate_resizable_append_input(xyz,varDict,fieldDict,ordering,point,ptable,nopartition):
	r'''
	Validate local input and MPI-wide schema for a resizable append.

	Args:
		xyz (np.ndarray): local point coordinates.
		varDict (dict): variables in the append block.
		fieldDict (dict): fields in the append block.
		ordering (np.ndarray): global ordering of local points.
		point (bool): whether the fields contain point data.
		ptable (PartitionTable): active partition table.
		nopartition (bool): whether global point ordering is stored directly.

	Returns:
		int: common append block size.
	'''
	if ptable is None:
		raiseError('Resizable append requires a partition table!')
	xyz      = np.asarray(xyz)
	ordering = np.asarray(ordering)
	if not xyz.ndim == 2:
		raiseError('Resizable append coordinates must be a two-dimensional array!')
	if not ordering.ndim == 1 or not ordering.shape[0] == xyz.shape[0]:
		raiseError('Resizable append ordering must contain one entry per point!')
	if not np.issubdtype(ordering.dtype,np.integer) or np.any(ordering < 0):
		raiseError('Resizable append ordering must contain non-negative integers!')
	if h5_resizable_append_npoints(xyz,ordering,ptable,nopartition) < 1:
		raiseError('Resizable append requires at least one spatial point!')
	block_size = h5_resizable_append_block_size(varDict,fieldDict)
	master     = ptable.has_master and MPI_RANK == 0
	if not master:
		for var in sorted(fieldDict.keys()):
			ndim = int(fieldDict[var]['ndim'])
			if point and ndim > 1:
				raiseError('Resizable append point fields must be one-dimensional!')
			if nopartition and ndim > 1:
				raiseError('Cannot deal with multi-dimensional arrays in no partition mode!')
			if not np.asarray(fieldDict[var]['value']).shape[0] == ndim*ordering.shape[0]:
				raiseError('Resizable append field <%s> has an invalid spatial size!'%var)
			if not nopartition:
				istart,iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=point)
				if not iend-istart == np.asarray(fieldDict[var]['value']).shape[0]:
					raiseError('Resizable append field <%s> does not match its partition bounds!'%var)
	# Every rank must create and resize the same HDF5 schema in the same order.
	schema = (
		(xyz.shape[1],xyz.dtype.str,bool(ptable.has_master)),
		tuple((var,int(varDict[var]['idim']),np.asarray(varDict[var]['value']).dtype.str,np.asarray(varDict[var]['value']).shape[0]) for var in sorted(varDict.keys())),
		tuple((var,int(fieldDict[var]['ndim']),np.asarray(fieldDict[var]['value']).dtype.str,np.asarray(fieldDict[var]['value']).shape[1]) for var in sorted(fieldDict.keys())),
		bool(point),bool(nopartition),
	)
	if MPI_SIZE > 1:
		schemas = MPI_COMM.allgather(schema)
		if not all(candidate == schemas[0] for candidate in schemas):
			raiseError('Resizable append schema differs between MPI ranks!')
	return block_size


def h5_resizable_append_npoints(xyz,ordering,ptable,nopartition):
	r'''
	Return the global spatial size represented by an append input block.

	Args:
		xyz (np.ndarray): local coordinates.
		ordering (np.ndarray): global ordering of local points.
		ptable (PartitionTable): active partition table.
		nopartition (bool): whether repeated points are removed by global index.

	Returns:
		int: global number of spatial points.
	'''
	if nopartition:
		local_max = int(np.max(ordering)) if ordering.shape[0] > 0 else -1
		return int(mpi_reduce(local_max,op='max',all=True)) + 1
	npoints = mpi_reduce(xyz.shape[0] if not np.any(np.isnan(xyz)) else 0,op='sum',all=True)
	if ptable.has_master: npoints -= 1
	return int(npoints)


def h5_validate_resizable_append_group(file,xyz,varDict,fieldDict,ordering,point,ptable,nopartition,block_size,layout_hash):
	r'''
	Validate an existing resizable append group and return cursor and capacity.

	The validation prevents appending incompatible names, dtypes, dimensions,
	partition layout, or spatial sizes to an existing file.

	Args:
		file (h5py.Group): ``DATASET`` group in the HDF5 file.
		xyz, varDict, fieldDict, ordering, point, ptable, nopartition: append input.
		block_size (int): validated input block size.
		layout_hash (np.ndarray or None): partitioned spatial layout fingerprint.

	Returns:
		(int, int): committed cursor and current physical capacity.
	'''
	cursor = h5_resizable_append_cursor(file)
	if cursor is None:
		raiseError('Existing DATASET was not created with append_resizable=True!')
	for attr in [H5_APPEND_BLOCK_SIZE,H5_APPEND_NOPARTITION]:
		if not attr in file.attrs: raiseError('Resizable append attribute <%s> is missing!'%attr)
	if not int(file.attrs[H5_APPEND_BLOCK_SIZE]) == block_size:
		raiseError('Resizable append block size differs from the first block!')
	if not bool(file.attrs[H5_APPEND_NOPARTITION]) == bool(nopartition):
		raiseError('Cannot change nopartition while appending to a resizable dataset!')
	if not nopartition:
		if not H5_APPEND_LAYOUT_HASH in file.attrs:
			raiseError('Resizable append spatial layout fingerprint is missing!')
		if not np.array_equal(np.asarray(file.attrs[H5_APPEND_LAYOUT_HASH]),layout_hash):
			raiseError('Resizable append spatial layout differs from the first block!')
	if not bool(int(file['pointData'][0])) == bool(point):
		raiseError('Cannot change point data type while appending to a resizable dataset!')
	if not int(file['npoints'][0]) == h5_resizable_append_npoints(xyz,ordering,ptable,nopartition):
		raiseError('Resizable append spatial size differs from the first block!')
	if not file['xyz'].shape[1] == xyz.shape[1]:
		raiseError('Resizable append coordinate dimension differs from the first block!')
	if not 'VARIABLES' in file or not 'FIELDS' in file:
		raiseError('Resizable append VARIABLES or FIELDS group is missing!')
	if not sorted(file['VARIABLES'].keys()) == sorted(varDict.keys()):
		raiseError('Resizable append variable names differ from the first block!')
	if not sorted(file['FIELDS'].keys()) == sorted(fieldDict.keys()):
		raiseError('Resizable append field names differ from the first block!')
	capacities = []
	for var in sorted(varDict.keys()):
		vargroup = file['VARIABLES'][var]
		value    = np.asarray(varDict[var]['value'])
		if not int(vargroup['idim'][0]) == int(varDict[var]['idim']):
			raiseError('Resizable append variable <%s> changed idim!'%var)
		if not vargroup['value'].dtype == value.dtype:
			raiseError('Resizable append variable <%s> changed dtype!'%var)
		if not vargroup['value'].ndim == 1 or not vargroup['value'].maxshape == (None,) or vargroup['value'].chunks is None:
			raiseError('Resizable append variable <%s> is not resizable!'%var)
		capacities.append(vargroup['value'].shape[0])
	for var in sorted(fieldDict.keys()):
		fieldgroup = file['FIELDS'][var]
		value      = np.asarray(fieldDict[var]['value'])
		ndim       = int(fieldDict[var]['ndim'])
		nrows      = ndim*int(file['npoints'][0])
		if not int(fieldgroup['ndim'][0]) == ndim or not int(fieldgroup['nvar'][0]) == 1:
			raiseError('Resizable append field <%s> changed dimensions!'%var)
		if not fieldgroup['vars'].shape == (1,) or not fieldgroup['value'].dtype == value.dtype:
			raiseError('Resizable append field <%s> has an incompatible schema!'%var)
		if not fieldgroup['value'].shape[0] == nrows or not fieldgroup['value'].maxshape == (nrows,None) or fieldgroup['value'].chunks is None:
			raiseError('Resizable append field <%s> is not resizable!'%var)
		capacities.append(fieldgroup['value'].shape[1])
	if len(set(capacities)) != 1:
		raiseError('Resizable append arrays have inconsistent capacities!')
	if cursor > capacities[0]:
		raiseError('Resizable append cursor exceeds the physical capacity!')
	return cursor,capacities[0]


def h5_resize_resizable_append_datasets(file,varDict,fieldDict,capacity):
	r'''
	Resize every append value array to a common physical capacity.

	This function must be called by every MPI rank because extending an HDF5
	dataset is a collective metadata operation.

	Args:
		file (h5py.Group): ``DATASET`` group in the HDF5 file.
		varDict (dict): variables in the append block.
		fieldDict (dict): fields in the append block.
		capacity (int): new physical capacity.
	'''
	for var in sorted(varDict.keys()):
		file['VARIABLES'][var]['value'].resize((capacity,))
	for var in sorted(fieldDict.keys()):
		value = file['FIELDS'][var]['value']
		value.resize((value.shape[0],capacity))


def h5_fill_resizable_append_datasets(file,varDict,fieldDict,ordering,point,ptable,nopartition,cursor,end):
	r'''
	Write one block into already sized resizable HDF5 arrays.

	Variables are replicated and written by rank zero. Field rows are written by
	the owning rank in partition mode or mapped through their global ordering in
	``nopartition`` mode.

	Args:
		file (h5py.Group): ``DATASET`` group in the HDF5 file.
		varDict (dict): variables in the append block.
		fieldDict (dict): fields in the append block.
		ordering (np.ndarray): global ordering of local points.
		point (bool): whether fields contain point data.
		ptable (PartitionTable): active partition table.
		nopartition (bool): whether rows use their global ordering.
		cursor (int): first position to write.
		end (int): position immediately after the block.
	'''
	if MPI_RANK == 0:
		for var in sorted(varDict.keys()):
			file['VARIABLES'][var]['value'][cursor:end] = np.asarray(varDict[var]['value'])
	if ptable.has_master and MPI_RANK == 0: return
	for var in sorted(fieldDict.keys()):
		value = np.asarray(fieldDict[var]['value'])
		ndim  = int(fieldDict[var]['ndim'])
		dset  = file['FIELDS'][var]['value']
		if nopartition:
			inods,idx = np.unique(ordering,return_index=True)
			components = np.arange(ndim,dtype=np.int64)
			destination = (ndim*inods[:,None] + components[None,:]).reshape(-1)
			source      = (ndim*idx[:,None]   + components[None,:]).reshape(-1)
			dset[destination,cursor:end] = value[source,:]
		else:
			istart,iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=point)
			if not iend-istart == value.shape[0]:
				raiseError('Resizable append field <%s> does not match its partition bounds!'%var)
			dset[istart:iend,cursor:end] = value

def h5_load_fields_single(file,npoints,ptable,varDict,point):
	'''
	Load the fields inside the HDF5 file
	'''
	# Read variables
	cursor    = h5_resizable_append_cursor(file)
	fieldDict = {}
	for v in file['FIELDS'].keys():
		fieldgroup = file['FIELDS'][v]
		# Load point and dimensions
		ndim = int(fieldgroup['ndim'][0])
		dims = [ndim*npoints] + ([cursor] if not cursor is None else list(fieldgroup['vars']))
		# Now allocate output array
		value = np.zeros(dims,fieldgroup['value'].dtype)
		# Select which points to load
		if point:
			inods = ptable.partition_points(npoints,ndim=ndim)
		else:
			# Use the partition bounds to recover the array
			istart, iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=False)
			inods = np.arange(istart,iend,dtype=np.int32)
		# Read only the committed part of a preallocated append dataset
		sliced  = tuple([inods] + [np.s_[:dim] for dim in dims[1:]])
		value[:] = np.array(fieldgroup['value'][sliced])
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
	h5_save_points(group,xyz,ordering,ptable,point)
	# Store the variables
	h5_fill_variable_datasets(h5_create_variable_datasets(group,varDict,ptable),varDict)
	# Store the fields
	h5_fill_field_datasets(h5_create_field_datasets(group,fieldDict,ptable),fieldDict,ptable,point,None,None)
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


@cr('h5IO.append_dset_resizable')
def h5_append_dset_resizable(fname,xyz,varDict,fieldDict,ordering,point,ptable,append_total_size=None,mode='a',mpio=True,nopartition=False):
	r'''
	Append a fixed-size block to consolidated, resizable HDF5 arrays.

	This opt-in variant retains the same ``VARIABLES`` and ``FIELDS`` hierarchy
	as a regular save. A persistent cursor separates the logical written length
	from optional preallocated capacity, so reopening the file continues at the
	last committed position.

	Args:
		fname (str): HDF5 file name.
		xyz (np.ndarray): local point coordinates.
		varDict (dict): variables in this append block.
		fieldDict (dict): fields in this append block.
		ordering (np.ndarray): global ordering of local points.
		point (bool): whether fields contain point data.
		ptable (PartitionTable): active partition table.
		append_total_size (int, optional): initial or enlarged physical capacity.
			The append dimension remains unlimited.
		mode (str, optional): HDF5 opening mode (default ``'a'``).
		mpio (bool, optional): use parallel HDF5 for multi-rank execution.
		nopartition (bool, optional): store rows using their global ordering.

	Notes:
		Every block must have the same length, names, dtypes, and dimensions as
		the first. Point fields and ``nopartition`` fields currently require
		``ndim=1``. Serial HDF5 cannot safely be used by multiple MPI ranks and
		is therefore rejected for this append mode. With MPI ``nopartition``,
		repeated global point IDs must contain identical coordinates and field
		values on every rank that owns them. Files with reserved capacity must be
		loaded by a pyLOM version that understands the persistent append cursor.
	'''
	if not mpio and MPI_SIZE > 1:
		raiseError('Resizable append requires mpio=True when using multiple MPI ranks!')
	block_size = h5_validate_resizable_append_input(xyz,varDict,fieldDict,ordering,point,ptable,nopartition)
	capacity   = h5_resizable_append_capacity(block_size,append_total_size)
	if MPI_SIZE > 1:
		capacities = MPI_COMM.allgather(capacity)
		if not all(candidate == capacities[0] for candidate in capacities):
			raiseError('append_total_size differs between MPI ranks!')
	if mpio and MPI_SIZE > 1:
		h5_append_dset_resizable_mpio(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable,nopartition,block_size,capacity)
	else:
		h5_append_dset_resizable_serial(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable,nopartition,block_size,capacity)


def h5_append_dset_resizable_serial(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable,nopartition,block_size,requested_capacity):
	r'''
	Open a file with the serial HDF5 driver and append one resizable block.

	This is the serial backend of :func:`h5_append_dset_resizable`; callers
	should normally use that dispatch function instead.

	Args:
		fname, mode, xyz, varDict, fieldDict, ordering, point, ptable,
			nopartition: see :func:`h5_append_dset_resizable`.
		block_size (int): validated append block length.
		requested_capacity (int): requested physical capacity.
	'''
	file = h5py.File(fname,mode)
	h5_append_dset_resizable_file(file,xyz,varDict,fieldDict,ordering,point,ptable,nopartition,block_size,requested_capacity)
	file.close()


def h5_append_dset_resizable_mpio(fname,mode,xyz,varDict,fieldDict,ordering,point,ptable,nopartition,block_size,requested_capacity):
	r'''
	Collectively open a file with parallel HDF5 and append one block.

	All ranks enter the same metadata operations while raw field data is written
	to each rank's spatial rows.

	Args:
		fname, mode, xyz, varDict, fieldDict, ordering, point, ptable,
			nopartition: see :func:`h5_append_dset_resizable`.
		block_size (int): validated append block length.
		requested_capacity (int): requested physical capacity.
	'''
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM)
	h5_append_dset_resizable_file(file,xyz,varDict,fieldDict,ordering,point,ptable,nopartition,block_size,requested_capacity)
	file.close()


def h5_append_dset_resizable_file(file,xyz,varDict,fieldDict,ordering,point,ptable,nopartition,block_size,requested_capacity):
	r'''
	Create or extend resizable append arrays in an open HDF5 file.

	The cursor attribute is committed only after the data and logical ``vars``
	lengths have been flushed. If a write is interrupted earlier, a subsequent
	append overwrites the uncommitted region rather than skipping it.

	Args:
		file (h5py.File): open serial or parallel HDF5 file.
		xyz, varDict, fieldDict, ordering, point, ptable, nopartition: see
			:func:`h5_append_dset_resizable`.
		block_size (int): validated append block length.
		requested_capacity (int): requested physical capacity.
	'''
	new_dataset = not 'DATASET' in file
	layout_hash = None if nopartition else h5_resizable_append_layout_hash(xyz,ordering)
	if not 'Version' in file.attrs:
		if not new_dataset: raiseError('HDF5 file version is missing!')
		file.attrs['Version'] = PYLOM_H5_VERSION
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	if new_dataset:
		group = file.create_group('DATASET')
		if nopartition:
			h5_save_points_nopartition(group,xyz,ordering,ptable,point)
		else:
			h5_save_points(group,xyz,ordering,ptable,point)
		group.attrs[H5_APPEND_MODE]        = np.bytes_(H5_RESIZABLE_APPEND)
		group.attrs[H5_APPEND_CURSOR]      = np.int64(0)
		group.attrs[H5_APPEND_BLOCK_SIZE]  = np.int64(block_size)
		group.attrs[H5_APPEND_NOPARTITION] = np.uint8(bool(nopartition))
		if not nopartition: group.attrs[H5_APPEND_LAYOUT_HASH] = layout_hash
		h5_create_resizable_variable_datasets(group,varDict,requested_capacity,block_size)
		h5_create_resizable_field_datasets(group,fieldDict,requested_capacity,block_size)
	group = file['DATASET']
	cursor,capacity = h5_validate_resizable_append_group(group,xyz,varDict,fieldDict,ordering,point,ptable,nopartition,block_size,layout_hash)
	end             = cursor + block_size
	capacity_new    = max(capacity,requested_capacity,end)
	if capacity_new > capacity:
		h5_resize_resizable_append_datasets(group,varDict,fieldDict,capacity_new)
	# Raw data writes may be independent; all metadata operations remain collective.
	h5_fill_resizable_append_datasets(group,varDict,fieldDict,ordering,point,ptable,nopartition,cursor,end)
	if MPI_SIZE > 1: MPI_COMM.Barrier()
	file.flush()
	if MPI_RANK == 0:
		for var in sorted(fieldDict.keys()):
			group['FIELDS'][var]['vars'][0] = end
	if MPI_SIZE > 1: MPI_COMM.Barrier()
	file.flush()
	# Commit the authoritative cursor last. Attribute updates are collective.
	group.attrs.modify(H5_APPEND_CURSOR,np.int64(end))
	file.flush()


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
	group.attrs['NOPARTITION'] = True 
	group.attrs['PARTS']       = ptable.n_partitions
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
	group.attrs['NOPARTITION'] = nopartition 
	group.attrs['PARTS']       = ptable.n_partitions
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
	nopartition = group.attrs.get('NOPARTITION',True)
	if nopartition and not ptable.check_split():
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

@cr('h5IO.save_QR')
def h5_save_QR(fname,Q,Y,B,ptable,nvars=1,pointData=True,mode='w'):
	r'''
	Store QR variables into an HDF5 file. Can be appended to another HDF by setting the mode to 'a'. Then no partition table will be saved.

	Args:
		fname (string): file name
		Q (np.ndarray): Q matrix
		Y (np.ndarray): Randomized matrix before QR
		B (np.ndarray): R upper triangular matrix or B in case of doing randomized QR
		ptable (PartitionTable): partition table
		nvars (int, optional): number of variables analyzed jointly (default=1)
		pointData(bool, optional): whether is point data or cell data (default=True)
		Writing mode (string, optional): the h5 will be written again or appended.
	'''
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM) if not MPI_SIZE == 1 else h5py.File(fname,mode)
	# Store attributes and partition table
	if not mode == 'a':
		file.attrs['Version'] = PYLOM_H5_VERSION
		# Store partition table
		h5_save_partition(file,ptable)
	# Now create a QR group
	group = file.create_group('QR')
	# Create the datasets for U, S and V
	group.create_dataset('pointData',(1,),dtype='u1',data=pointData)
	group.create_dataset('n_variables',(1,),dtype='u1',data=nvars)
	Qsize = (mpi_reduce(Q.shape[0],op='sum',all=True),Q.shape[1]) if Q is not None else None
	Ysize = (mpi_reduce(Y.shape[0],op='sum',all=True),Y.shape[1]) if Y is not None else None
	dsetQ = group.create_dataset('Q',Qsize,dtype=Q.dtype)         if Q is not None else None
	dsetY = group.create_dataset('Y',Ysize,dtype=Y.dtype)         if Y is not None else None
	dsetB = group.create_dataset('B',B.shape,dtype=B.dtype)       if B is not None else None
	# Store S and U that are repeated across the ranks
	# So it is enough that one rank stores them
	if is_rank_or_serial(0):
		if dsetB is not None: dsetB[:] = B 
	# Store U in parallel
	istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,points=pointData)
	if dsetQ is not None: dsetQ[istart:iend,:] = Q
	if dsetY is not None: dsetY[istart:iend,:] = Y
	file.close()

@cr('h5IO.load_QR')
def h5_load_QR(fname,vars,ptable=None):
	r'''
	Load QR variables from an HDF5 file.
	
	Args:
		fname (string): file name
		vars (list): variables to load, it must be any of Q, B, Y.
		ptable (PartitionTable, optional): partition table used to load the data (default, None)
	
	Returns:
		list: list of the np.ndarray requested to load.
	'''
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM) if not MPI_SIZE == 1 else h5py.File(fname,'r')
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Read the requested variables S, V
	varList = []
	if 'Q' in vars:
		# Check if we need to read the partition table
		if ptable is None: ptable = h5_load_partition(file)
		# Read
		nvars = int(file['QR']['n_variables'][0])
		point = bool(file['QR']['pointData'][0])
		istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,points=point)
		varList.append(np.array(file['QR']['Q'][istart:iend,:]))
	if 'Y' in vars:
		# Check if we need to read the partition table
		if ptable is None: ptable = h5_load_partition(file)
		# Read
		nvars = int(file['QR']['n_variables'][0])
		point = bool(file['QR']['pointData'][0])
		istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,points=point)
		varList.append(np.array(file['QR']['Y'][istart:iend,:]))
	if 'B' in vars: varList.append( np.array(file['QR']['B'][:,:]))
	# Return
	file.close()
	return varList

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
	Usize = (mpi_reduce(U.shape[0],op='sum',all=True),U.shape[1]) if U is not None else None
	dsetU = group.create_dataset('U',Usize,dtype=U.dtype)   if U is not None else None
	dsetS = group.create_dataset('S',S.shape,dtype=S.dtype) if S is not None else None
	dsetV = group.create_dataset('V',V.shape,dtype=V.dtype) if V is not None else None
	# Store S and U that are repeated across the ranks
	# So it is enough that one rank stores them
	if is_rank_or_serial(0):
		if dsetS is not None: dsetS[:] = S
		if dsetV is not None: dsetV[:] = V 
	# Store U in parallel
	istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,points=pointData)
	if dsetU is not None: dsetU[istart:iend,:] = U
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
		varList.append(np.array(file['POD']['U'][istart:iend,:]) if nmod < 0 else np.array(file['POD']['U'][istart:iend,:nmod]))
	if 'S' in vars: varList.append( np.array(file['POD']['S'][:]) if nmod < 0 else  np.array(file['POD']['S'][:nmod]) )
	if 'V' in vars: varList.append( np.array(file['POD']['V'][:,:]) if nmod < 0 else np.array(file['POD']['V'][:nmod,:]) )
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
		istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,points=point)
		varList.append( np.array(file['DMD']['Phi'][istart:iend,:]) )
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

@cr('io.create_compressed')
def h5_create_compressed(fname:str,basedir:str,r:int,nmod:int,nvars:int,nlayers:int,conv_chan:int,kernel:int,nAEsG:int,nptxAE:int,dtype:np.dtype):
	r'''
	Function to create the groups of the decoders, the latent spaces and the scalers when the Q matrix from a randomized QR factorization is compressed using GAVI:

	AFEGIR LA CITA DELS PROCEEDINGS DE MADRID

	Args:
		fname (str): name of the file
		basedir (str): folder in which the file will be saved
		r (int): truncation modes of the latent space
		nmod (int): number of modes being compressed
		nvars (int): number of variables being compressed
		nlayers (int): number of convolutional layers in the decoder
		conv_channels (int): number of convolutional channels that has each layer of the decoder
		kernel (int): kernel size of the convolutions
		nAEsG (int): total number of autoencoders used to compress the matrix
		nptxAE (int): number of points per autoencoder in the matrix
		dtype (np.dtype): precision in which to save the arrays

	Returns
		h5py.File the file is not closed and once this function creates it, it returns its pointer to be used during the compression. 
		Note: The file must be closed at the end of the compression using file.close()

	'''
	file  = h5py.File('%s/%s.h5' % (basedir, fname), mode="w", driver='mpio', comm=MPI_COMM)
	stats = file.create_group("STATS")
	stats.create_dataset("mean", shape=(nAEsG,nvars), dtype=dtype)
	stats.create_dataset("std",  shape=(nAEsG,nvars), dtype=dtype)
	decod = file.create_group("DECODER")
	decod.create_dataset("weights", shape=(nAEsG,conv_chan,nvars,kernel), dtype=dtype)
	decod.create_dataset("biases", shape=(nAEsG,nvars), dtype=dtype)
	lats  = file.create_group("LATENTS")
	lats.create_dataset("Q",  shape=(nAEsG,int(nmod/2**nlayers)*conv_chan,r), dtype=dtype)
	lats.create_dataset("B", shape=(nAEsG,r,nptxAE), dtype=dtype)

	file.close()

@cr('io.flush_compressed')
def h5_flush_compressed(fname:str,basedir:str,ist:int,ien:int,means:np.ndarray,stds:np.ndarray,weights:np.ndarray,biases:np.ndarray,Q:np.ndarray,B:np.ndarray):
	r'''
	Function to save the data into the hdf5 file created with the h5_create_compressed function so that at every compression iteration the scalers, decoder parameters and the factorization of the latent space are properly saved

	AFEGIR LA CITA DELS PROCEEDINGS DE MADRID

	Args:
		fname (str): file in which the data has to be saved
		basedir (str): folder in which the file will be saved
		ist (int): ID of the first element to be compressed by the current core
		ien (int): ID of the last element
		means (np.ndarray): array containing the mean of the compressed data
		stds (np.ndarray): array containing the std of the compressed data
		weights (np.ndarray): array containing the weights of the decoders
		biases (np.ndarray): array containing the biases of the decoders
		Q (np.ndarray): Q matrix of the factorization of the latent vectors
		B (np.ndarray): B matrix of the factorization of the latent vectors
		r (int): truncation value of the factorized latent vectors

	Returns;
		h5py.File file in which the data has been saved. It must be closed when all cores finish compressing their data
	'''
	file  = h5py.File('%s/%s.h5' % (basedir, fname), mode="a", driver='mpio', comm=MPI_COMM)
	file['STATS/mean'][ist:ien,:] = means
	file['STATS/std'][ist:ien,:]  = stds
	file['DECODER/weights'][ist:ien,:,:,:] = weights
	file['DECODER/biases'][ist:ien,:]      = biases
	file['LATENTS/Q'][ist:ien,:,:]  = Q
	file['LATENTS/B'][ist:ien,:,:] = B

	file.close()
	

@cr('io.load_compressed')
def h5_load_compressed(fname:str, basedir:str, ptable:PartitionTable, nelxAE:int):
	r"""
	Load the necessary information to decompress the Q matrix atre using GAVI:
	
	CITA PROCEEDINGS MADRID
	
	Args:
		fname (str): name of the file to load
		basedir (str): directory where the file is located
		ptable (PartitionTable): partition of the mesh in which the data will be represented after decompression:
		nelxAE (int): number of elements in each autoencoder
		
	Returns:
		[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] The 6 arrays saved in the compressed file: the mean and standard deviation of the inputs, the decoder parameters and the factorized latent spaces	
	"""
	ist,ien = ptable.partition_bounds(MPI_RANK,points=False)
	ist,ien = int(ist/nelxAE), int(ien/nelxAE)
	file    = h5py.File('%s/%s' % (basedir,fname), mode="r", driver='mpio', comm=MPI_COMM)
	Qmeans  = np.array(file['/STATS/mean'][ist:ien,:])
	Qstds   = np.array(file['/STATS/std'][ist:ien,:])
	weights = np.array(file['/DECODER/weights'][ist:ien,:,:,:])
	biases  = np.array(file['/DECODER/biases'][ist:ien,:])
	Q       = np.array(file['/LATENTS/Q'][ist:ien,:,:])
	B       = np.array(file['/LATENTS/B'][ist:ien,:,:])
	file.close()

	return Qmeans, Qstds, weights, biases, Q, B


def _h5_prep_numeric_features(features_dict):
	"""
	Validate graph feature dictionary and cast numeric arrays to float32.
	"""
	out = {}
	for name, arr in features_dict.items():
		a = np.asarray(arr)
		if a.dtype == np.dtype('O') or a.dtype.kind in ('U', 'S'):
			raiseError(f"Feature '{name}' has non-numeric dtype={a.dtype}. Move it to METADATA or drop it.")
		if a.dtype.kind not in ('f', 'c'):
			a = a.astype('float32', copy=False)
		else:
			a = a.astype('float32', copy=False)
		out[name] = a
	return out

def _h5_decode_bytes_list(values):
	"""
	Decode a list of bytes or strings to a list of strings. This is used to decode the feature names stored as byte strings in HDF5 attributes.
	Args:
		values: list of bytes or strings
	Returns:
		list of strings
	"""
	out = []
	for v in values:
		if isinstance(v, (bytes, bytearray)):
			out.append(v.decode('utf8'))
		else:
			out.append(str(v))
	return out


@cr('io.save_graph_serial')
def h5_save_graph_serial(
	fname,
	num_nodes,
	num_edges,
	edge_index,
	node_features_dict,
	edge_features_dict,
	mode='w',
):
	"""
	Save a Graph in HDF5 (serial mode), strict flat schema with ordering.

	Schema
	------
	/GRAPH
	  attrs['schema']     = "graph_flat_v2"
	  numNodes            : i4[1]
	  numEdges            : i4[1]
	  edgeIndex           : i4[2,E]
	  NODEFEATRS (group)
		attrs['feature_names'] : S[]
		<feat_name>            : float32[N, k_i]
	  EDGEFEATRS (group)
		attrs['feature_names'] : S[]
		<feat_name>            : float32[E, k_i]
	"""

	if node_features_dict is None or edge_features_dict is None:
		raiseError("Both node and edge feature dictionaries are required.")

	# Normalize edge_index shape and dtype -> int32 on disk
	edge_index = np.asarray(edge_index)
	if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
		edge_index = edge_index.T  # Ensure shape (2, E)
	edge_index = edge_index.astype('int32', copy=False)

	# Validate that features are numeric and cast to float32
	node_features_dict = _h5_prep_numeric_features(node_features_dict)
	edge_features_dict = _h5_prep_numeric_features(edge_features_dict)

	f  = h5py.File(fname, mode=mode)
	f.attrs['Version'] = PYLOM_H5_VERSION

	if 'GRAPH' in f:
		raiseError("/GRAPH group already exists. Use a new output file or mode='w'.")
	g = f.create_group('GRAPH')
	g.attrs['schema'] = 'graph_flat_v2'  # Store a str attribute for version validation

	g.create_dataset('numNodes', (1,), dtype='i4', data=int(num_nodes))
	g.create_dataset('numEdges', (1,), dtype='i4', data=int(num_edges))
	g.create_dataset('edgeIndex', data=edge_index, dtype='i4')

	# Node features
	node_grp = g.create_group('NODEFEATRS')
	node_names = list(node_features_dict.keys())
	node_grp.attrs['feature_names'] = np.array(node_names, dtype='S')

	for name in node_names:
		node_grp.create_dataset(name, data=node_features_dict[name])

	# Edge features
	edge_grp = g.create_group('EDGEFEATRS')
	edge_names = list(edge_features_dict.keys())
	edge_grp.attrs['feature_names'] = np.array(edge_names, dtype='S')

	for name in edge_names:
		edge_grp.create_dataset(name, data=edge_features_dict[name])

	f.close()


@cr('io.load_graph_serial')
def h5_load_graph_serial(fname):
	"""
	Load a Graph from HDF5 (serial mode), strict flat schema.

	Returns
	-------
	num_nodes : int
	num_edges : int
	edge_index : np.ndarray, shape (2, E), dtype=int64
	node_features_dict : OrderedDict[str, np.ndarray]  # float32 arrays
	edge_features_dict : OrderedDict[str, np.ndarray]  # float32 arrays
	"""
	f  = h5py.File(fname, mode='r')
	if 'GRAPH' not in f:
		raiseError("Missing /GRAPH group in HDF5 file.")
	g = f['GRAPH']

	# Strict schema check
	schema = g.attrs.get('schema', None)
	if schema is None:
		raiseError("Missing /GRAPH.attrs['schema']. Expected 'graph_flat_v2'.")
	if isinstance(schema, (bytes, bytearray)):
		schema = schema.decode('utf8')
	if str(schema) != 'graph_flat_v2':
		raiseError(f"Unsupported graph schema '{schema}'. Expected 'graph_flat_v2'.")

	num_nodes = int(np.array(g['numNodes'])[0])
	num_edges = int(np.array(g['numEdges'])[0])

	edge_index = np.array(g['edgeIndex'])
	# Normalize to (2, E) int64 for in-memory usage
	if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
		edge_index = edge_index.T
	edge_index = edge_index.astype('int64', copy=False)

	# Node features in stored order
	if 'NODEFEATRS' not in g:
		raiseError("Missing /GRAPH/NODEFEATRS group.")
	node_grp = g['NODEFEATRS']
	if 'feature_names' not in node_grp.attrs:
		raiseError("Missing feature_names attribute in /GRAPH/NODEFEATRS.")

	node_names = _h5_decode_bytes_list(node_grp.attrs['feature_names'])
	node_features_dict = OrderedDict()
	for name in node_names:
		arr = np.array(node_grp[name])
		# Enforce numeric and float32
		if arr.dtype == np.dtype('O') or arr.dtype.kind in ('U', 'S'):
			raiseError(f"Node feature '{name}' has non-numeric dtype={arr.dtype}.")
		if arr.dtype.kind not in ('f', 'c'):
			arr = arr.astype('float32', copy=False)
		else:
			arr = arr.astype('float32', copy=False)
		node_features_dict[name] = arr

	# Edge features in stored order
	if 'EDGEFEATRS' not in g:
		raiseError("Missing /GRAPH/EDGEFEATRS group.")
	edge_grp = g['EDGEFEATRS']
	if 'feature_names' not in edge_grp.attrs:
		raiseError("Missing feature_names attribute in /GRAPH/EDGEFEATRS.")

	edge_names = _h5_decode_bytes_list(edge_grp.attrs['feature_names'])
	edge_features_dict = OrderedDict()
	for name in edge_names:
		arr = np.array(edge_grp[name])
		if arr.dtype == np.dtype('O') or arr.dtype.kind in ('U', 'S'):
			raiseError(f"Edge feature '{name}' has non-numeric dtype={arr.dtype}.")
		if arr.dtype.kind not in ('f', 'c'):
			arr = arr.astype('float32', copy=False)
		else:
			arr = arr.astype('float32', copy=False)
		edge_features_dict[name] = arr

	f.close()

	return num_nodes, num_edges, edge_index, node_features_dict, edge_features_dict
