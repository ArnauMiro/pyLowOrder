#!/usr/bin/env python
#
# pyLOM, IO
#
# H5 Input Output
#
# Last rev: 31/07/2021
from __future__ import print_function, division

import numpy as np, h5py

from ..utils.parall import MPI_COMM, MPI_RANK, MPI_SIZE, worksplit, writesplit, is_rank_or_serial, mpi_reduce
from ..utils.errors import raiseError
from ..utils.mesh   import STRUCT2D, STRUCT3D, UNSTRUCT, mesh_number_of_points


def h5_save(fname,xyz,time,pointOrder,cellOrder,meshDict,varDict,mpio=True,write_master=True):
	'''
	Save a Dataset in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		h5_save_mpio(fname,xyz,time,pointOrder,cellOrder,meshDict,varDict,write_master)
	else:
		h5_save_serial(fname,xyz,time,pointOrder,cellOrder,meshDict,varDict)

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

def h5_save_serial(fname,xyz,time,pointOrder,cellOrder,meshDict,varDict):
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
	# Store ordering arrays
	dset = file.create_dataset('pointOrder',pointOrder.shape,dtype=pointOrder.dtype,data=pointOrder)
	dset = file.create_dataset('cellOrder',cellOrder.shape,dtype=cellOrder.dtype,data=cellOrder)
	# Store the DATA
	data_group = file.create_group('DATA')
	for var in varDict.keys():
		h5_save_variable_serial(data_group,var,varDict[var])
	file.close()

def h5_dataset_variable_mpio(group,varname,varDict,nnod,ncells):
	'''
	Save a variable inside an HDF5 group
	'''
	npoints   = nnod if varDict['point'] else ncells
	var_group = group.create_group(varname)
	dset = var_group.create_dataset('point',(1,),dtype=int,data=varDict['point'])
	dset = var_group.create_dataset('ndim',(1,),dtype=int,data=varDict['ndim'])
	dset = var_group.create_dataset('value',(varDict['ndim']*npoints,varDict['value'].shape[1]),dtype=varDict['value'].dtype)
	return dset

def h5_save_mpio(fname,xyz,time,pointOrder,cellOrder,meshDict,varDict,write_master):
	'''
	Save a dataset in HDF5 in parallel mode
	'''
	# Open file
	file = h5py.File(fname,'w',driver='mpio',comm=MPI_COMM)
	dsetDict = {}
	# Compute the total number of points
	npoints = mpi_reduce(pointOrder.shape[0],op='sum',all=True)
	ncells  = mpi_reduce(cellOrder.shape[0],op='sum',all=True)
	# Create datasets
	# number of points and number of instants
	dset = file.create_dataset('npoints',(1,),dtype='i',data=npoints)
	dset = file.create_dataset('ninstants',(1,),dtype='i',data=time.shape[0])
	# time instants
	dset = file.create_dataset('time',time.shape,dtype=time.dtype,data=time)
	# xyz coordinates
	dsetDict['xyz'] = file.create_dataset('xyz',(npoints,xyz.shape[1]),dtype=xyz.dtype)
	# ordering arrays
	dsetDict['pointOrder'] = file.create_dataset('pointOrder',(npoints,),dtype=pointOrder.dtype)
	dsetDict['cellOrder']  = file.create_dataset('cellOrder',(ncells,),dtype=cellOrder.dtype)
	# DATA group
	data_group = file.create_group('DATA')
	for var in varDict.keys():
		dsetDict[var] = h5_dataset_variable_mpio(data_group,var,varDict[var],npoints,ncells)
	# Store datasets
	if MPI_RANK != 0 or write_master:
		# Obtain the write split for point and cell arrays
		istart_p, iend_p = writesplit(pointOrder.shape[0],write_master)
		istart_c, iend_c = writesplit(cellOrder.shape[0],write_master)
		# Store xyz coordinates
		dsetDict['xyz'][istart_p:iend_p,:] = xyz
		# Store ordering arrays
		dsetDict['pointOrder'][istart_p:iend_p] = pointOrder
		dsetDict['cellOrder'][istart_c:iend_c]  = cellOrder
		# Store the DATA
		for var in varDict.keys():
			v = varDict[var]
			if v['point']:
				dsetDict[var][v['ndim']*istart_p:v['ndim']*iend_p,:] = v['value']
			else:
				dsetDict[var][v['ndim']*istart_c:v['ndim']*iend_c,:] = v['value']
	file.close()
	# Append mesh in serial mode
	if is_rank_or_serial(0):
		file = h5py.File(fname,'a')
		# Create a group to store the mesh details
		mesh_group = file.create_group('MESH')
		h5_save_mesh(mesh_group,meshDict)
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
	# Load ordering arrays
	pointOrder = np.array(file['pointOrder'],dtype=np.double)
	cellOrder  = np.array(file['cellOrder'],dtype=np.double)
	# Load the variables in the varDict
	varDict = {}
	for var in file['DATA'].keys():
		varDict[var] = h5_load_variable_serial(file['DATA'][var])
	file.close()
	return xyz, time, pointOrder, cellOrder, meshDict, varDict

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
		# Load ordering arrays
		pointOrder = np.array(file['pointOrder'][istart:iend],dtype=np.double)
		cellOrder  = np.array(file['cellOrder'][istart:iend],dtype=np.double)
		# Load the variables in the varDict
		varDict = {}
		for var in file['DATA'].keys():
			varDict[var] = h5_load_variable_mpio(file['DATA'][var],meshDict)
	else:
		raiseError('H5IO not implemented!')
	file.close()
	return xyz, time, pointOrder, cellOrder, meshDict, varDict
