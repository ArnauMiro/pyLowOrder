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
from ..mesh            import MTYPE2ID, ID2MTYPE, Mesh
from ..utils.cr        import cr
from ..utils.parall    import MPI_COMM, MPI_RANK, MPI_SIZE, worksplit, writesplit, is_rank_or_serial, mpi_reduce, mpi_gather
from ..utils.errors    import raiseError


PYLOM_H5_VERSION = (2,0)


@cr('h5IO.save')
def h5_save(fname,time,varDict,mesh,ptable,mpio=True,nopartition=False):
	'''
	Save a Dataset in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		h5_save_mpio(fname,time,varDict,mesh,ptable,nopartition)
	else:
		h5_save_serial(fname,time,varDict,mesh,ptable)

@cr('h5IO.append')
def h5_append(fname,time,varDict,mesh,ptable,mpio=True,nopartition=False):
	'''
	Save a Dataset in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		h5_append_mpio(fname,time,varDict,mesh,ptable,nopartition)
	else:
		h5_append_serial(fname,time,varDict,mesh,ptable)

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
		dset = group.create_dataset('type',(1,),dtype='i4',data=MTYPE2ID[mesh.type])
		# Write the total number of cells and the total number of points
		# Assume we might be dealing with a parallel mesh
		npointG, ncellG = mesh.npointsG, mesh.ncellsG
		if ptable.has_master: 
			npointG -= 1
			ncellG  -= 1
		group.create_dataset('npoints',(1,),dtype='i4',data=npointG)
		group.create_dataset('ncells' ,(1,),dtype='i4',data=ncellG)
		# Create the rest of the datasets for parallel storage
		dxyz   = group.create_dataset('xyz',(npointG,mesh.ndim),dtype='f8')
		dconec = group.create_dataset('connectivity',(ncellG,mesh.nnodcell),dtype='i4')
		deltyp = group.create_dataset('eltype',(ncellG,),dtype='u1')
		dcellO = group.create_dataset('cellOrder',(ncellG,),dtype='i4')
		dpoinO = group.create_dataset('pointOrder',(npointG,),dtype='i4')
		# Skip master if needed
		if ptable.has_master and MPI_RANK == 0: return None, None, None
		# Point dataset
		# Compute start and end of read, node data
		istartp, iend = ptable.partition_bounds(MPI_RANK,points=True)
		dxyz[istartp:iend,:]  = mesh.xyz
		dpoinO[istartp:iend]  = mesh.pointOrder
		# Compute start and end of read, cell data
		istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
		dconec[istart:iend,:] = mesh.connectivity + istartp
		deltyp[istart:iend]   = mesh.eltype
		dcellO[istart:iend]   = mesh.cellOrder
	return None, None, None

def h5_save_mesh_nopartition(file,mesh,ptable):
	'''
	Save the mesh inside the HDF5 file
	'''
	# Skip the whole process if the mesh is not there
	if mesh is not None:
		# Create a group for the mesh
		group = file.create_group('MESH')
		# Save the mesh type
		dset = group.create_dataset('type',(1,),dtype='i4',data=MTYPE2ID[mesh.type])
		# Write the total number of cells and the total number of points
		# Assume we might be dealing with a parallel mesh
		npointG, ncellG = mesh.npointsG2, mesh.ncellsG2
		group.create_dataset('npoints',(1,),dtype='i4',data=npointG)
		group.create_dataset('ncells' ,(1,),dtype='i4',data=ncellG)
		# Create the rest of the datasets for parallel storage
		dxyz   = group.create_dataset('xyz',(npointG,mesh.ndim),dtype='f8')
		dconec = group.create_dataset('connectivity',(ncellG,mesh.nnodcell),dtype='i4')
		deltyp = group.create_dataset('eltype',(ncellG,),dtype='u1')
		dcellO = group.create_dataset('cellOrder',(ncellG,),dtype='i4')
		dpoinO = group.create_dataset('pointOrder',(npointG,),dtype='i4')
		# Skip master if needed
		if ptable.has_master and MPI_RANK == 0: return None, None, None
		# Get the position where the points should be stored
		inods,idx = np.unique(mesh.pointOrder,return_index=True)
		# Write dataset - points
		dxyz[inods,:] = mesh.xyz[idx,:]
		dpoinO[inods] = mesh.pointOrder[idx]
                # Compute start and end of read, cell data
		istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
		# Write dataset - cells
		dconec[istart:iend,:] = mesh.pointOrder[mesh.connectivity]
		deltyp[istart:iend]   = mesh.eltype
		dcellO[istart:iend]   = mesh.cellOrder
	return inods,idx,npointG

def h5_create_variable_datasets(file,time,varDict,ptable,ipart=-1):
	'''
	Create the variable datasets inside an HDF5 file
	'''
	# Store time array (common for all processes)
	if not 'time' in file.keys(): file.create_dataset('time',time.shape,dtype=time.dtype,data=time)
	# Create group for variables
	group = file.create_group('VARIABLES_%d'%ipart if ipart >= 0 else 'VARIABLES')
	dsetDict = {}
	for var in varDict.keys():
		vargroup = group.create_group(var)
		n     = mpi_reduce(varDict[var]['value'].shape[0],op='sum',all=True)
		if ptable.has_master: n -= 1
		npoin = int(file['MESH']['npoints'][0]) if varDict[var]['point'] else int(file['MESH']['ncells'][0])
		ndim  = n//npoin
		ntime = varDict[var]['value'].shape[1]
		dsetDict[var] = {
			'point' : vargroup.create_dataset('point',(1,),dtype='u1'),
			'ndim'  : vargroup.create_dataset('ndim' ,(1,),dtype='i4'),
			'value' : vargroup.create_dataset('value',(ndim*npoin,ntime),dtype=varDict[var]['value'].dtype),
		}
	return dsetDict

def h5_fill_variable_datasets(dsetDict,varDict,ptable,inods,idx):
	'''
	Fill in the variable datasets inside an HDF5 file
	'''
	# Skip master if needed
	if ptable.has_master and MPI_RANK == 0: return
	for var in dsetDict.keys():
		# Fill dataset
		dsetDict[var]['point'][:] = varDict[var]['point']
		dsetDict[var]['ndim'][:]  = varDict[var]['ndim']
		if inods is None or not varDict[var]['point']:
			# Compute start and end bounds for the variable
			istart, iend = ptable.partition_bounds(MPI_RANK,ndim=varDict[var]['ndim'],points=varDict[var]['point'])
			dsetDict[var]['value'][istart:iend,:]  = varDict[var]['value']
		else:
			if varDict[var]['ndim'] > 1: raiseError('Cannot deal with multi-dimensional arrays in no partition mode!')
			dsetDict[var]['value'][inods,:] = varDict[var]['value'][idx,:]

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
	inods,idx,npoints = h5_save_mesh(file,mesh,ptable)
	# Store the variables
	h5_fill_variable_datasets(h5_create_variable_datasets(file,time,varDict,ptable),varDict,ptable,inods,idx)
	file.close()

def h5_save_mpio(fname,time,varDict,mesh,ptable,nopartition):
	'''
	Save a dataset in HDF5 in parallel mode
	'''
	# Open file
	file = h5py.File(fname,'w',driver='mpio',comm=MPI_COMM)
	file.attrs['Version'] = PYLOM_H5_VERSION
	# Store partition table
	h5_save_partition(file,PartitionTable.new(1,mesh.ncellsG2,mesh.npointsG2))
	# Store the mesh
	inods,idx,npoints = h5_save_mesh(file,mesh,ptable) if not nopartition else h5_save_mesh_nopartition(file,mesh,ptable)
	# Store the variables
	h5_fill_variable_datasets(h5_create_variable_datasets(file,time,varDict,ptable),varDict,ptable,inods,idx)
	file.close()

def h5_append_serial(fname,time,varDict,mesh,ptable):
	'''
	Save a dataset in HDF5 in serial mode
	'''
	file = h5py.File(fname,'a')
	if not hasattr(h5_append_serial,'ipart'):
		# Input file does not exist, we create it with the whole structure
		file.attrs['Version'] = PYLOM_H5_VERSION
		# Store partition table
		h5_save_partition(file,ptable)
		# Store the mesh
		inods,idx,npoints = h5_save_mesh(file,mesh,ptable)
		# Start the partition counter
		h5_append_serial.ipart   = 0
		h5_append_serial.inods   = inods
		h5_append_serial.idx     = idx
		h5_append_serial.npoints = npoints
	else:
		# Check the file version
		version = tuple(file.attrs['Version'])
		if not version == PYLOM_H5_VERSION:
			raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
		# Obtain from function
		ipart   = h5_append_serial.ipart
		inods   = h5_append_serial.inods
		idx     = h5_append_serial.idx
		npoints = h5_append_serial.npoints 
		# Write the time partition on the file
		h5_fill_variable_datasets(h5_create_variable_datasets(file,time,varDict,ptable,ipart=ipart),varDict,ptable,inods,idx)
		# Update time
		file['time'][:] = time
		# Increase the partition counter
		h5_append_serial.ipart += 1
	file.close()

def h5_append_mpio(fname,time,varDict,mesh,ptable,nopartition):
	'''
	Save a dataset in HDF5 in serial mode
	'''
	file = h5py.File(fname,'a',driver='mpio',comm=MPI_COMM)
	if not hasattr(h5_append_mpio,'ipart'):
		# Input file does not exist, we create it with the whole structure
		file.attrs['Version'] = PYLOM_H5_VERSION
		# Store partition table
		h5_save_partition(file,ptable)
		# Store the mesh
		inods,idx,npoints = h5_save_mesh(file,mesh,ptable) if not nopartition else h5_save_mesh_nopartition(file,mesh,ptable)
		# Start the partition counter
		h5_append_mpio.ipart   = 0
		h5_append_mpio.inods   = inods
		h5_append_mpio.idx     = idx
		h5_append_mpio.npoints = npoints
	else:
		# Check the file version
		version = tuple(file.attrs['Version'])
		if not version == PYLOM_H5_VERSION:
			raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
		# Obtain from function
		ipart   = h5_append_mpio.ipart
		inods   = h5_append_mpio.inods
		idx     = h5_append_mpio.idx
		npoints = h5_append_mpio.npoints 
		# Write the time partition on the file
		h5_fill_variable_datasets(h5_create_variable_datasets(file,time,varDict,ptable,ipart=ipart),varDict,ptable,inods,idx)
		# Update time
		file['time'][:] = time
		# Increase the partition counter
		h5_append_mpio.ipart += 1
	file.close()


@cr('h5IO.load')
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

def h5_load_mesh(file,ptable,repart):
	'''
	Load the mesh inside the HDF5 file
	'''
	if not 'MESH' in file.keys(): return None
	# Read mesh type
	mtype  = ID2MTYPE[int(file['MESH']['type'][0])]
	# Read cell related variables
	istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
	conec  = np.array(file['MESH']['connectivity'][istart:iend,:],np.int32)
	eltype = np.array(file['MESH']['eltype'][istart:iend],np.int32) 
	cellO  = np.array(file['MESH']['cellOrder'][istart:iend],np.int32)
	# Read point related variables
	if repart:
		# Warning! Repartition will only work if the input file is serial
		# i.e., it does not have any repeated nodes, otherwise it wont work
		inods  = ptable.partition_points(MPI_RANK,1,conec)
		ptable.update_points(inods.shape[0])
	else:
		istart, iend = ptable.partition_bounds(MPI_RANK,points=True)
		inods = np.arange(istart,iend,dtype=np.int32)
	xyz    = np.array(file['MESH']['xyz'][inods,:],np.double) 
	pointO = np.array(file['MESH']['pointOrder'][inods],np.int32)
	# Fix the connectivity to start at zero
	conec = np.searchsorted(pointO,conec.flatten()).reshape(conec.shape).astype(np.int32)
	# Return
	return Mesh(mtype,xyz,conec,eltype,cellO,pointO),inods

def h5_load_variables_single(file,mesh,ptable,inods,repart):
	'''
	Load the variables inside the HDF5 file
	'''
	# Read time
	time = np.array(file['time'][:])
	# Read variables
	varDict = {}
	for v in file['VARIABLES'].keys():
		# Load point and ndim
		point   = bool(file['VARIABLES'][v]['point'][0])
		ndim    = int(file['VARIABLES'][v]['ndim'][0])
		npoints = mesh.npoints if point else mesh.ncells
		value   = np.zeros((ndim*npoints,len(time)),np.double) 
		# Read the values
		if mesh is None or not point:
			istart, iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=point)
			value[:,:]   = np.array(file['VARIABLES'][v]['value'][istart:iend,:])
		else:
			if repart:
				# We are repartitioning, then use inods to read the array
				for idim in range(ndim):
					value[idim:ndim*npoints:ndim,:] = np.array(file['VARIABLES'][v]['value'][inods+idim*npoints,:])
			else:
				# Just use the partition bounds to recover the array
				istart, iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=point)
				value[:,:]   = np.array(file['VARIABLES'][v]['value'][istart:iend,:])
		# Generate dictionary
		varDict[v] = {'point':point,'ndim':ndim,'value':value}
	# Return
	return time, varDict

def h5_load_variables_multi(file,mesh,ptable,inods,repart,npart):
	'''
	Load the variables inside the HDF5 file
	'''
	# Read time
	time = np.array(file['time'][:])
	# Scan for variables in first partition and build variable dictionary
	varDict = {}
	for v in file['VARIABLES_0'].keys():
		point   = bool(file['VARIABLES_0'][v]['point'][0])
		ndim    = int(file['VARIABLES_0'][v]['ndim'][0])
		npoints = mesh.npoints if point else mesh.ncells
		value   = np.zeros((ndim*npoints,len(time)),np.double) 		
		# Generate dictionary
		varDict[v] = {'point':point,'ndim':ndim,'value':value}
	# Read variables per partition
	psize  = len(time)//npart
	for ipart in range(npart):
		# Compute start and end of my partition in time
		pstart = ipart*psize
		pend   = (ipart+1)*psize
		pname  = 'VARIABLES_%d'%ipart
		# Read the partition
		for v in file[pname].keys():
			point   = bool(file[pname][v]['point'][0])
			ndim    = int(file[pname][v]['ndim'][0])
			npoints = mesh.npoints if point else mesh.ncells
			# Read the values
			if mesh is None or not point:
				istart, iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=point)
				varDict[v]['value'][:,pstart:pend] = np.array(file[pname][v]['value'][istart:iend,:])
			else:
				if repart:
					# We are repartitioning, then use inods to read the array
					for idim in range(ndim):
						varDict[v]['value'][idim:ndim*npoints:ndim,pstart:pend] = np.array(file[pname][v]['value'][inods+idim*npoints,:])
				else:
					# Just use the partition bounds to recover the array
					istart, iend = ptable.partition_bounds(MPI_RANK,ndim=ndim,points=point)
					varDict[v]['value'][:,pstart:pend] = np.array(file[pname][v]['value'][istart:iend,:])
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
	repart = False
	# Are we reading for the same number of partitions?
	if not ptable.check_split():
		# Read the number of elements and points to compute
		# the new partition table
		npoints, ncells = h5_load_size(file)
		# Redo the partitions table
		ptable = PartitionTable.new(MPI_SIZE,ncells,npoints)
		repart = True
	# Read the mesh
	mesh, inods = h5_load_mesh(file,ptable,repart)
	# Figure out how many partitions we have
	npart = np.sum(['VAR' in key for key in file.keys()])
	# Read the variables
	time, varDict = h5_load_variables_single(file,mesh,ptable,inods,repart) if npart == 1 else h5_load_variables_multi(file,mesh,ptable,inods,repart,npart)
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
	repart = False
	# Are we reading for the same number of partitions?
	if not ptable.check_split():
		# Read the number of elements and points to compute
		# the new partition table
		npoints, ncells = h5_load_size(file)
		# Redo the partitions table
		ptable = PartitionTable.new(MPI_SIZE,ncells,npoints)
		repart = True
	# Read the mesh
	mesh, inods = h5_load_mesh(file,ptable,repart)
	# Figure out how many partitions we have
	npart = np.sum(['VAR' in key for key in file.keys()])
	# Read the variables
	time, varDict = h5_load_variables_single(file,mesh,ptable,inods,repart) if npart == 1 else h5_load_variables_multi(file,mesh,ptable,inods,repart,npart)
	file.close()
	return ptable, mesh, time, varDict


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
		istart, iend = ptable.partition_bounds(MPI_RANK,ndim=nvars,point=point)
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