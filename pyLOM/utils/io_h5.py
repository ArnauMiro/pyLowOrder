#!/usr/bin/env python
#
# H5 Input Output
#
# Last rev: 19/02/2021
from __future__ import print_function, division

import numpy as np, h5py, mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from .errors import raiseError


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
		h5_save_serial(fname,xyz,time,meshDict,varDict,metadata)

def h5_save_mesh(group,meshDict):
	'''
	Save the meshDict inside the HDF5 group
	'''
	pass

def h5_save_serial(fname,xyz,time,meshDict,varDict):
	'''
	Save a field in HDF5 in serial mode
	'''
	# Open file for writing
	file = h5py.File(fname,'w')
	# Create a group to store the mesh details
	mesh_group = file.create_group('MESH')
	h5_save_mesh(mesh_group,meshDict)
	# Store number of points and number of instants
	dset    = file.create_dataset('npoints',(1,),dtype='i')
	dset[:] = xyz.shape[0]
	dset    = file.create_dataset('ninstants',(1,),dtype='i')
	dset[:] = time.shape[0]
	# Store xyz coordinates
	dset      = file.create_dataset('xyz',xyz.shape,dtype=xyz.dtype)
	dset[:,:] = xyz
	# Store time instants
	dset    = file.create_dataset('time',time.shape,dtype=time.dtype)
	dset[:] = time
	# Store the DATA
	data_group = file.create_group('DATA')
	for var in varDict.keys():
		dset      = data_group.create_dataset(var,varDict[var].shape,dtype=varDict[var].dtype)
		dset[:,:] = varDict[var]
	file.close()

def h5_save_mpio(fname,xyz,time,meshDict,varDict,write_master=False):
	'''
	Save a field in HDF5 in parallel mode
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

#def h5_load_field(fname,mpio=True):
#	'''
#	Load a field in HDF5
#	'''
#	if mpio and not MPIsize == 1:
#		return h5_load_field_mpio(fname)
#	else:
#		return h5_load_field_serial(fname)
#
#def h5_load_field_serial(fname):
#	'''
#	Load a field in HDF5 in serial
#	'''
#	# Open file for reading
#	file = h5py.File(fname,'r')
#	# Load node coordinates
#	xyz     = np.array(file['xyz'],dtype=np.double)
#	varDict = {}
#	# Load the variables in the varDict
#	for var in file.keys():
#		if var == 'xyz':      continue # Skip xyz
#		if var == 'metadata': continue # Skip metadata
#		varDict[var] = np.array(file[var],dtype=np.double)
#	file.close()
#	return xyz, varDict
#
#def h5_load_field_mpio(fname):
#	'''
#	Load a field in HDF5 in parallel
#	'''
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