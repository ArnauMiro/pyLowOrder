#!/usr/bin/env python
#
# pyLOM, IO
#
# H5 Input Output
#
# Last rev: 31/07/2021
from __future__ import print_function, division

import numpy as np, h5py

from ..utils.parall import MPI_COMM, MPI_RANK, MPI_SIZE, worksplit, is_rank_or_serial, mpi_reduce
from ..utils.errors import raiseError
from ..utils.mesh   import STRUCT2D, STRUCT3D, UNSTRUCT, mesh_number_of_points


def h5_save(fname,xyz,time,meshDict,varDict,**kwargs):
	'''
	Save a Dataset in HDF5
	'''
	if not 'mpio' in kwargs.keys(): kwargs['mpio'] = True
	if kwargs['mpio'] and not MPI_SIZE == 1:
		if not 'pointOrder' in kwargs.keys() or not 'cellOrder' in kwargs.keys():
			raiseError('H5IO mpio error! ordering arrays not provided!')
		h5_save_mpio(fname,xyz,time,meshDict,varDict,kwargs['pointOrder'],kwargs['cellOrder'])
	else:
		h5_save_serial(fname,xyz,time,meshDict,varDict)

def h5_save_mesh(group,meshDict):
	'''
	Save the meshDict inside the HDF5 group
	'''
	# Save the mesh type
	dset = group.create_dataset('type',(1,),dtype=h5py.special_dtype(vlen=str),data=meshDict['type'])
	# Save mesh data according to the type
	if meshDict['type'].lower() in STRUCT2D:
		# 2D structured mesh, store nx and ny
		dset = group.create_dataset('nx',(1,),dtype=int,data=meshDict['nx'])
		dset = group.create_dataset('ny',(1,),dtype=int,data=meshDict['ny'])
	if meshDict['type'].lower() in STRUCT3D:
		# 3D structured mesh, store nx, ny and nz
		dset = group.create_dataset('nx',(1,),dtype=int,data=meshDict['nx'])
		dset = group.create_dataset('ny',(1,),dtype=int,data=meshDict['ny'])		
		dset = group.create_dataset('nz',(1,),dtype=int,data=meshDict['nz'])		
	if meshDict['type'].lower() in UNSTRUCT:
		# Unstructured mesh, store nel, element kind (elkind) and connectivity (conec)
		dset = group.create_dataset('nnod',(1,),dtype=int,data=meshDict['nnod'])
		dset = group.create_dataset('nel',(1,),dtype=int,data=meshDict['nel'])
		dset = group.create_dataset('elkind',(1,),dtype=h5py.special_dtype(vlen=str),data=meshDict['elkind'])
		dset = group.create_dataset('conec',conec.shape,dtype=conec.dtype,data=meshDict['conec'])
	if 'partition' in meshDict.keys():
		raiseError('Not implemented!')

def h5_save_variable_serial(group,varname,varDict):
	'''
	Save a variable inside an HDF5 group
	'''
	var_group = group.create_group(varname)
	dset = var_group.create_dataset('point',(1,),dtype=int,data=varDict['point'])
	dset = var_group.create_dataset('ndim',(1,),dtype=int,data=varDict['ndim'])
	dset = var_group.create_dataset('value',varDict['value'].shape,dtype=varDict['value'].dtype,data=varDict['value'])

def h5_save_serial(fname,xyz,time,meshDict,varDict):
	'''
	Save a dataset in HDF5 in serial mode
	'''
	# Open file for writing
	file = h5py.File(fname,'w')
	# Create a group to store the mesh details
	mesh_group = file.create_group('MESH')
	h5_save_mesh(mesh_group,meshDict)
	# Store number of points and number of instants
	dset = file.create_dataset('npoints',(1,),dtype='i',data=xyz.shape[0])
	dset = file.create_dataset('ninstants',(1,),dtype='i',data=time.shape[0])
	# Store xyz coordinates
	dset = file.create_dataset('xyz',xyz.shape,dtype=xyz.dtype,data=xyz)
	# Store time instants
	dset = file.create_dataset('time',time.shape,dtype=time.dtype,data=time)
	# Store the DATA
	data_group = file.create_group('DATA')
	for var in varDict.keys():
		h5_save_variable_serial(data_group,var,varDict[var])
	file.close()

def h5_save_variable_mpio(group,varname,varDict,pointOrder,cellOrder):
	'''
	Save a variable inside an HDF5 group
	'''
	var_group = group.create_group(varname)
	dset = var_group.create_dataset('point',(1,),dtype=int,data=varDict['point'])
	dset = var_group.create_dataset('ndim',(1,),dtype=int,data=varDict['ndim'])
	dset = var_group.create_dataset('value',varDict['value'].shape,dtype=varDict['value'].dtype)
	dset[pointOrder if varDict['point'] else cellOrder,:] = varDict['value']

def h5_save_mpio(fname,xyz,time,meshDict,varDict,pointOrder,cellOrder):
	'''
	Save a dataset in HDF5 in parallel mode
	'''
	# Open file
	file = h5py.File(fname,'w',driver='mpio',comm=comm)
	# Compute the total number of points
	npoints = mpi_reduce(xyz.shape[0],op='sum',all=True)
	# Create a group to store the mesh details
	mesh_group = file.create_group('MESH')
	h5_save_mesh(mesh_group,meshDict)
	# Store number of points and number of instants
	dset = file.create_dataset('npoints',(1,),dtype='i',data=npoints)
	dset = file.create_dataset('ninstants',(1,),dtype='i',data=time.shape[0])
	# Store xyz coordinates
	dset = file.create_dataset('xyz',xyz.shape,dtype=xyz.dtype)
	dset[pointOrder,:] = xyz
	# Store time instants
	dset = file.create_dataset('time',time.shape,dtype=time.dtype,data=time)
	# Store the DATA
	data_group = file.create_group('DATA')
	for var in varDict.keys():
		h5_save_variable_mpio(data_group,var,varDict[var],pointOrder,cellOrder)
	file.close()


def h5_load(fname,mpio=True):
	'''
	Load a dataset in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		return h5_load_mpio(fname)
	else:
		return h5_load_serial(fname)

def h5_load_mesh(group):
	'''
	Load the meshDict inside the HDF5 group
	'''
	meshDict = {}
	# Load the mesh type
	meshDict['type'] = group['type'][0].decode('utf-8')
	# Save mesh data according to the type
	if meshDict['type'].lower() in STRUCT2D:
		# 2D structured mesh, load nx and ny
		meshDict['nx'] = group['nx'][0]
		meshDict['ny'] = group['ny'][0]
	if meshDict['type'].lower() in STRUCT3D:
		# 3D structured mesh, load nx, ny and nz
		meshDict['nx'] = group['nx'][0]
		meshDict['ny'] = group['ny'][0]
		meshDict['nz'] = group['nz'][0]
	if meshDict['type'].lower() in UNSTRUCT:
		# Unstructured mesh, store nel, element kind (elkind) and connectivity (conec)
		meshDict['nnod']   = group['nnod'][0]
		meshDict['nel']    = group['nel'][0]
		meshDict['elkind'] = group['elkind'][0].decode('utf-8')
#		meshDict['conec']  = np.array(group['conec'],dtype=np.int32)
	return meshDict

def h5_load_variable_serial(group):
	'''
	Save a variable inside an HDF5 group
	'''
	varDict = {
		'point' : group['point'][0],
		'ndim'  : group['ndim'][0],
		'value' : np.array(group['value'],dtype=np.double)
	}
	return varDict

def h5_load_serial(fname):
	'''
	Load a dataset in HDF5 in serial
	'''
	# Open file for reading
	file = h5py.File(fname,'r')
	# Load mesh details
	meshDict = h5_load_mesh(file['MESH'])
	# Load node coordinates
	xyz  = np.array(file['xyz'],dtype=np.double)
	# Load time instants
	time = np.array(file['time'],dtype=np.double)
	# Load the variables in the varDict
	varDict = {}
	for var in file['DATA'].keys():
		varDict[var] = h5_load_variable_serial(file['DATA'][var])
	file.close()
	return xyz, time, meshDict, varDict

def h5_load_variable_mpio(group,meshDict):
	'''
	Save a variable inside an HDF5 group
	'''
	# Read variable metadata
	varDict = {
		'point' : group['point'][0],
		'ndim'  : group['ndim'][0]
	}
	# Compute the number of points per variable
	npoints = varDict['ndim']*mesh_number_of_points(varDict['point'],meshDict)
	# Call the worksplit and only read a part of the data
	istart,iend = worksplit(0,npoints,MPI_RANK,nWorkers=MPI_SIZE)
	varDict['value'] = np.array(group['value'][istart:iend,:],dtype=np.double)
	return varDict

def h5_load_mpio(fname):
	'''
	Load a field in HDF5 in parallel
	'''
	# Open file for reading
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM)
	# Load mesh details
	meshDict = h5_load_mesh(file['MESH'])
	# Load time instants
	time = np.array(file['time'],dtype=np.double)
	# If we do not have information on the partition stored in
	# the file, generate a simple partition
	if not 'partition' in file.keys():
		# Read the number of points of the mesh
		npoints = int(file['npoints'][0])
		# Call the worksplit and only read a part of the data
		istart,iend = worksplit(0,npoints,MPI_RANK,nWorkers=MPI_SIZE)
		# Load node coordinates
		xyz = np.array(file['xyz'][istart:iend,:],dtype=np.double)
		# Load the variables in the varDict
		varDict = {}
		for var in file['DATA'].keys():
			varDict[var] = h5_load_variable_mpio(file['DATA'][var],meshDict)
	else:
		raiseError('H5IO not implemented!')
	file.close()
	return xyz, time, meshDict, varDict
