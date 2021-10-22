#!/usr/bin/env python
#
# pyLOM, IO
#
# H5 Input Output
#
# Last rev: 31/07/2021
from __future__ import print_function, division

import numpy as np, h5py, mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from ..utils.errors import raiseError
from ..utils.mesh   import STRUCT2D, STRUCT3D, UNSTRUCT


comm    = MPI.COMM_WORLD
rank    = comm.Get_rank()
MPIsize = comm.Get_size()


def h5_save(fname,xyz,time,meshDict,varDict,mpio=True,write_master=False):
	'''
	Save a Dataset in HDF5
	'''
	if mpio and not MPIsize == 1:
		h5_save_mpio(fname,xyz,time,meshDict,varDict,write_master)
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
		dset = group.create_dataset('nx',(1,),dtype=int)
	if meshDict['type'].lower() in UNSTRUCT:
		# Unstructured mesh, store nel, element kind (elkind) and connectivity (conec)
		dset = group.create_dataset('nnod',(1,),dtype=int,data=meshDict['nnod'])
		dset = group.create_dataset('nel',(1,),dtype=int,data=meshDict['nel'])
		dset = group.create_dataset('elkind',(1,),dtype=h5py.special_dtype(vlen=str),data=meshDict['elkind'])
		dset = group.create_dataset('conec',conec.shape,dtype=conec.dtype,data=meshDict['conec'])
	if 'partition' in meshDict.keys():
		raiseError('Not implemented!')

def h5_save_variable(group,varname,varDict):
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
	dset    = file.create_dataset('npoints',(1,),dtype='i',data=xyz.shape[0])
	dset    = file.create_dataset('ninstants',(1,),dtype='i',data=time.shape[0])
	# Store xyz coordinates
	dset      = file.create_dataset('xyz',xyz.shape,dtype=xyz.dtype,data=xyz)
	# Store time instants
	dset    = file.create_dataset('time',time.shape,dtype=time.dtype,data=time)
	# Store the DATA
	data_group = file.create_group('DATA')
	for var in varDict.keys():
		h5_save_variable(data_group,var,varDict[var])
	file.close()

def h5_save_mpio(fname,xyz,time,meshDict,varDict,write_master=False):
	'''
	Save a dataset in HDF5 in parallel mode
	'''
	raiseError('Not implemented!')
#	# Compute the total number of points
#	npG  = int(comm.allreduce(xyz.shape[0] if not np.all(np.isnan(xyz)) else 0.,op=MPI.SUM))
#	# Open file
#	file = h5py.File(fname,'w',driver='mpio',comm=comm)
#	# Create groups and datasets
#	meta_group = file.create_group('metadata')
#	dset_meta = {}
#	dset_meta['npoints'] = meta_group.create_dataset('npoints',(1,),dtype='i')
#	for var in metadata.keys():
#		dset_meta[var] = meta_group.create_dataset(var,(1,),dtype=metadata[var][1])
#	dset_file = {}
#	dset_file['xyz'] = file.create_dataset('xyz',(npG,3),dtype='f')
#	for var in varDict.keys():
#		v = varDict[var]
#		dset_file[var] = file.create_dataset(var,(npG,) if len(v.shape) == 1 else (npG,v.shape[1]),dtype='f')
#	# Master writes the metadata
#	if rank == 0:
#		dset_meta['npoints'][:] = npG
#		for var in metadata.keys():
#			dset_meta[var][:] = metadata[var][0]
#	if rank != 0 or write_master:
#		rstart = 1 if not write_master else 0
#		# Select in which order the processors will write
#		if rank == rstart:
#			istart, iend = 0, xyz.shape[0] if not np.all(np.isnan(xyz)) else 0
#			comm.send(iend,dest=rank+1) # send to next where to start writing
#		elif rank == MPIsize-1:
#			istart = comm.recv(source=rank-1) # recive from the previous where to start writing
#			iend   = istart + xyz.shape[0] if not np.all(np.isnan(xyz)) else istart
#		else:
#			istart = comm.recv(source=rank-1) # recive from the previous where to start writing
#			iend   = istart + xyz.shape[0] if not np.all(np.isnan(xyz)) else istart
#			comm.send(iend,dest=rank+1) # send to next where to start writing
#		# Store the data
#		dset_file['xyz'][istart:iend,:] = xyz
#		for var in varDict.keys():
#			v = varDict[var]
#			if len(v.shape) == 1: # Scalar field
#				dset_file[var][istart:iend] = v
#			else: # Vectorial or tensorial field
#				dset_file[var][istart:iend,:] = v
#	file.close()


def h5_load(fname,mpio=True):
	'''
	Load a dataset in HDF5
	'''
	if mpio and not MPIsize == 1:
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
		meshDict['conec']  = np.array(group['conec'],dtype=np.int32)
	if 'partition' in group.keys():
		raiseError('Not implemented!')
	return meshDict

def h5_load_variable(group):
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
		varDict[var] = h5_load_variable(file['DATA'][var])
	file.close()
	return xyz, time, meshDict, varDict

def h5_load_mpio(fname):
	'''
	Load a field in HDF5 in parallel
	'''
	raiseError('Not implemented!')
#	# Open file for reading
#	file = h5py.File(fname,'r',driver='mpio',comm=comm)
#	# Read the number of points
#	npoints = int(file['metadata']['npoints'][0])
#	# Call the worksplit and only read a part of the data
#	istart, iend = worksplit(0,npoints)
#	# Load node coordinates
#	xyz     = np.array(file['xyz'][istart:iend,:],dtype=np.double)
#	varDict = {}
#	# Load the variables in the varDict
#	for var in file.keys():
#		if var == 'xyz':      continue # Skip xyz
#		if var == 'metadata': continue # Skip metadata
#		if len(file[var].shape) == 1:
#			# Scalar field
#			varDict[var] = np.array(file[var][istart:iend],dtype=np.double)
#		else:
#			# Vectorial field
#			varDict[var] = np.array(file[var][istart:iend,:],dtype=np.double)
#	file.close()
#	return xyz, varDict