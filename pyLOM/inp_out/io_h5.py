#!/usr/bin/env python
#
# pyLOM, IO
#
# H5 Input Output
#
# Last rev: 31/07/2021
from __future__ import print_function, division

import numpy as np, h5py

from ..partition_table import PartitionTable
from ..mesh            import Mesh
from ..utils.parall    import MPI_COMM, MPI_RANK, MPI_SIZE, worksplit, writesplit, is_rank_or_serial, mpi_reduce, mpi_gather
from ..utils.errors    import raiseError


PYLOM_H5_VERSION = (2,0)


def h5_save(fname,time,varDict,mesh,ptable,mpio=True):
	'''
	Save a Dataset in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		h5_save_mpio(fname,time,varDict,mesh,ptable)
	else:
		h5_save_serial(fname,time,varDict,mesh,ptable)

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

def h5_save_mesh(file,mesh,ptable):
	'''
	Save the mesh inside the HDF5 file
	'''
	# Skip the whole process if the mesh is not there
	if mesh is not None:
		# Create a group for the mesh
		group = file.create_group('MESH')
		# Save the mesh type
		dset = group.create_dataset('type',(1,),dtype=h5py.special_dtype(vlen=str),data=mesh.type)
		# Write the total number of cells and the total number of points
		# Assume we might be dealing with a parallel mesh
		npointG, ncellG = mesh.npointsG, mesh.ncellsG
		group.create_dataset('npoints',(1,),dtype='i4',data=npointG)
		group.create_dataset('ncells' ,(1,),dtype='i4',data=ncellG)
		# Create the rest of the datasets for parallel storage
		dxyz   = group.create_dataset('xyz',(npointG,mesh.ndim),dtype='f8')
		dconec = group.create_dataset('connectivity',(ncellG,mesh.nnodcell),dtype='i4')
		deltyp = group.create_dataset('eltype',(ncellG,),dtype='u1')
		dcellO = group.create_dataset('cellOrder',(ncellG,),dtype='i4')
		dpoinO = group.create_dataset('pointOrder',(npointG,),dtype='i4')
		# Skip master if needed
		if ptable.has_master and MPI_RANK == 0: return
		# Point dataset
		# Compute start and end of read, node data
		istart, iend = ptable.partition_bounds(MPI_RANK,points=True)
		dxyz[istart:iend,:]   = mesh.xyz
		dpoinO[istart:iend]   = mesh.pointOrder
		# Compute start and end of read, cell data
		istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
		dconec[istart:iend,:] = mesh.connectivity
		deltyp[istart:iend]   = mesh.eltype
		dcellO[istart:iend]   = mesh.cellOrder

def h5_create_variable_datasets(file,time,varDict):
	'''
	Create the variable datasets inside an HDF5 file
	'''
	# Store time array (common for all processes)
	file.create_dataset('time',time.shape,dtype=time.dtype,data=time)
	# Create group for variables
	group = file.create_group('VARIABLES')
	dsetDict = {}
	for var in varDict.keys():
		vargroup = group.create_group(var)
		size  = (mpi_reduce(varDict[var]['value'].shape[0],op='sum',all=True), time.shape[0])
		dsetDict[var] = {
			'point' : vargroup.create_dataset('point',(1,),dtype='u1'),
			'ndim'  : vargroup.create_dataset('ndim' ,(1,),dtype='i4'),
			'value' : vargroup.create_dataset('value',size,dtype=varDict[var]['value'].dtype),
		}
	return dsetDict

def h5_fill_variable_datasets(dsetDict,varDict,ptable):
	'''
	Fill in the variable datasets inside an HDF5 file
	'''
	# Skip master if needed
	if ptable.has_master and MPI_RANK == 0: return
	for var in dsetDict.keys():
		# Compute start and end bounds for the variable
		istart, iend = ptable.partition_bounds(MPI_RANK,ndim=varDict[var]['ndim'],points=varDict[var]['point'])
		# Fill dataset
		dsetDict[var]['point'][:] = varDict[var]['point']
		dsetDict[var]['ndim'][:]  = varDict[var]['ndim']
		if varDict[var]['ndim'] > 1:
			dsetDict[var]['value'][istart:iend,:]  = varDict[var]['value']
		else:
			dsetDict[var]['value'][istart:iend]  = varDict[var]['value']

def h5_save_serial(fname,time,varDict,mesh,ptable):
	'''
	Save a dataset in HDF5 in serial mode
	'''
	# Open file for writing
	file = h5py.File(fname,'w')
	file.attrs['Version'] = PYLOM_H5_VERSION
	# Store partition table
	h5_save_partition(file,ptable)
	# Store the mesh
	h5_save_mesh(file,mesh,ptable)
	# Store the variables
	h5_fill_variable_datasets(h5_create_variable_datasets(file,time,varDict),varDict,ptable)
	file.close()

def h5_save_mpio(fname,xyz,time,varDict,mesh,ptable):
	'''
	Save a dataset in HDF5 in parallel mode
	'''
	# Open file
	file = h5py.File(fname,'w',driver='mpio',comm=MPI_COMM)
	file.attrs['Version'] = PYLOM_H5_VERSION
	# Store partition table
	h5_save_partition(file,ptable)
	# Store the mesh
	h5_save_mesh(file,mesh,ptable)
	# Store the variables
	h5_fill_variable_datasets(h5_create_variable_datasets(file,time,varDict),varDict,ptable)
	file.close()


def h5_load(fname,mpio=True):
	'''
	Load a dataset in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		return h5_load_mpio(fname)
	else:
		return h5_load_serial(fname)

def h5_load_partition(file):
	'''
	Load a partition table inside an HDF5 file
	'''
	# Load file
	nparts   = int(file['PARTITIONS']['NSubD'][0])
	ids      = np.array(file['PARTITIONS']['Ids'][:])
	elements = np.array(file['PARTITIONS']['Elements'][:])
	points   = np.array(file['PARTITIONS']['Points'][:])
	# Return partition class
	return PartitionTable(nparts,ids,elements,points)

def h5_load_size(file):
	'''
	Load only the number of cells and points for the partition
	'''
	# Crash if mesh is not present
	if not 'MESH' in file.keys():
		raiseError('Repartition is not possible without a mesh!')
	# If the mesh is present read the size
	npoints = int(file['MESH']['npoints'][0])
	ncells  = int(file['MESH']['ncells'][0])
	return npoints, ncells

def h5_load_mesh(file,ptable):
	'''
	Load the mesh inside the HDF5 file
	'''
	if not 'MESH' in file.keys(): return None
	# Read mesh type
	mtype  = [c.decode('utf-8') for c in file['MESH']['type'][:]][0]
	# Read cell related variables
	istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
	conec  = np.array(file['MESH']['connectivity'][istart:iend,:],np.int32)
	eltype = np.array(file['MESH']['eltype'][istart:iend],np.int32) 
	cellO  = np.array(file['MESH']['cellOrder'][istart:iend],np.int32)
	# Read point related variables
	inods  = ptable.partition_points(MPI_RANK,1,conec)
	ptable.update_points(inods.shape[0])
	xyz    = np.array(file['MESH']['xyz'][inods,:],np.double) 
	pointO = np.array(file['MESH']['pointOrder'][inods],np.int32)
	# Fix the connectivity to start at zero
	conec -= conec.min()
	# Return
	return Mesh(mtype,xyz,conec,eltype,cellO,pointO),inods

def h5_load_variables(file,mesh,ptable,inods):
	'''
	Load the variables inside the HDF5 file
	'''
	# Read time
	time = np.array(file['time'][:])
	# Read variables
	varDict = {}
	for v in file['VARIABLES'].keys():
		# Load point and ndim
		point = bool(file['VARIABLES'][v]['point'][0])
		ndim  = int(file['VARIABLES'][v]['ndim'][0])
		# Read the values
		if point:
			# Dealing with point data
			if mesh is None:
				istart, iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=True)
				value = np.array( file['VARIABLES'][v]['value'][istart:iend,:])
			else:
				value = np.array(file['VARIABLES'][v]['value'][inods,:])
		else:
			# Dealing with cell data
			istart, iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=False)
			value = np.array(file['VARIABLES'][v]['value'][istart:iend,:])
		# Generate dictionary
		varDict[v] = {'point':point,'ndim':ndim,'value':value}
	# Return
	return time, varDict

def h5_load_serial(fname):
	'''
	Load a dataset in HDF5 in serial
	'''
	# Open file for writing
	file = h5py.File(fname,'r')
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Read partition table
	ptable = h5_load_partition(file)
	# Read the mesh
	mesh, inods = h5_load_mesh(file,ptable)
	# Read the variables
	time, varDict = h5_load_variables(file,mesh,ptable,inods)
	file.close()
	return ptable, mesh, time, varDict

def h5_load_mpio(fname):
	'''
	Load a field in HDF5 in parallel
	'''
	# Open file for reading
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM)
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
	# Read partition table
	ptable = h5_load_partition(file)
	# Are we reading for the same number of partitions?
	if not ptable.n_partitions == MPI_SIZE:
		# Read the number of elements and points to compute
		# the new partition table
		npoints, ncells = h5_load_size(file)
		# Redo the partitions table
		ptable = PartitionTable.new(MPI_SIZE,ncells,npoints)
	# Read the mesh
	mesh, inods = h5_load_mesh(file,ptable)
	# Read the variables
	time, varDict = h5_load_variables(file,mesh,ptable,inods)
	file.close()
	return ptable, mesh, time, varDict


def h5_save_POD(fname,U,S,V,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store POD variables into an HDF5 file.
	Can be appended to another HDF by setting the
	mode to 'a'. Then no partition table will be saved.
	'''
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM)
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
	dsetU = group.create_dataset('U',Usize,dtype='f8')
	dsetS = group.create_dataset('S',S.shape,dtype='f8')
	dsetV = group.create_dataset('V',V.shape,dtype='f8')
	# Store S and U that are repeated across the ranks
	# So it is enough that one rank stores them
	if is_rank_or_serial(0):
		dsetS[:] = S
		dsetV[:] = V
	# Store U in parallel
	istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,points=pointData)
	dsetU[istart:iend,:] = U
	file.close()

def h5_load_POD(fname,vars,ptable=None):
	'''
	Load POD variables from an HDF5 file.
	'''
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM)
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
		istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,point=point)
		varList.append( np.array(file['POD']['U'][istart:iend,:]) )
	if 'S' in vars: varList.append( np.array(file['POD']['S'][:]) )
	if 'V' in vars: varList.append( np.array(file['POD']['V'][:,:]) )
	# Return
	file.close()
	return varList


def h5_save_DMD(fname,muReal,muImag,Phi,bJov,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store DMD variables into an HDF5 file.
	Can be appended to another HDF by setting the
	mode to 'a'. Then no partition table will be saved.
	'''
	file = h5py.File(fname,mode,driver='mpio',comm=MPI_COMM)
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
	dsMu  = group.create_dataset('Mu',(muReal.shape[0],2),dtype='f8')
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

def h5_load_DMD(fname,vars,ptable=None):
	'''
	Load DMD variables from an HDF5 file.
	'''
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM)
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
		varList.append( np.array(file['DMD']['Phi'][istart:iend,:]) )
	if 'mu' in vars: 
		varList.append( np.array(file['DMD']['Mu'][:,0]) ) # Real
		varList.append( np.array(file['DMD']['Mu'][:,1]) ) # Imag
	if 'bJov' in vars: varList.append( np.array(file['DMD']['bJov'][:,:]) )
	# Return
	file.close()
	return varList
