#!/usr/bin/env python
#
# VTKH5 Input Output
#
# Last rev: 28/10/2022
from __future__ import print_function, division

import numpy as np, h5py

from ..utils.cr     import cr
from ..utils.parall import MPI_RANK, MPI_SIZE, MPI_COMM, mpi_reduce, mpi_bcast

VTKTYPE = np.string_('UnstructuredGrid')
VTKVERS = np.array([1,0],np.int32)


def _vtkh5_create_structure(file):
	'''
	Create the basic structure of a VTKH5 file
	'''
	# Create main group
	main = file.create_group('VTKHDF')
	main.attrs['Type']    = VTKTYPE
	main.attrs['Version'] = VTKVERS
	# Create cell data group
	main.create_group('CellData')
	main.create_group('PointData')
	main.create_group('FieldData')
	# Return created groups
	return main

def _vtkh5_connectivity_and_offsets(lnods):
	'''
	Build the offsets array (starting point per each element)

	'''
	# Compute the number of points per cell
	ppcell = np.sum(lnods >= 0,axis=1)
	# Compute the number of zeros per cell
	zpcell = np.sum(lnods < 0,axis=1)
	# Flatten the connectivity array
	lnodsf = lnods.flatten('c')
	# Now build the offsets vector
	offset = np.zeros((ppcell.shape[0]+1,),np.int32)
	offset[1:] = np.cumsum(ppcell) + np.cumsum(zpcell)
	return lnodsf, offset

def _vtkh5_write_mesh_serial(file,xyz,lnods,ltype):
	'''
	Write the mesh and the connectivity to the VTKH5 file.
	'''
	# Create dataset for number of points
	npoints, ndim = xyz.shape
	file.create_dataset('NumberOfPoints',(1,),dtype=int,data=npoints)
	file.create_dataset('Points',(npoints,ndim),dtype=np.double,data=xyz)
	# Create dataset for number of cells
	lnods, offsets = _vtkh5_connectivity_and_offsets(lnods)
	ncells = ltype.shape[0]
	ncsize = lnods.shape[0]
	file.create_dataset('NumberOfCells',(1,),dtype=int,data=ncells)
	file.create_dataset('NumberOfConnectivityIds',(1,),dtype=int,data=ncsize)
	file.create_dataset('Connectivity',(ncsize,),dtype=int,data=lnods)
	file.create_dataset('Offsets',(ncells+1,),dtype=int,data=offsets)
	file.create_dataset('Types',(ncells,),dtype=np.uint8,data=ltype)
	# Return some parameters
	return npoints, ncells 

def _vtkh5_write_mesh_mpio(file,xyz,lnods,ltype, ptable):
	'''
	Write the mesh and the connectivity to the VTKH5 file.
	'''
	myrank, nparts  = MPI_RANK, MPI_SIZE
	# Create datasets for point data
	npoints      = xyz.shape[0]                               # Number of points of this partition
	npG          = int(mpi_reduce(npoints,op='sum',all=True)) # Total number of points
	npoints_dset = file.create_dataset('NumberOfPoints',(nparts,),dtype=int)
	points_dset  = file.create_dataset('Points',(npG,3),dtype=np.double)
	# Create datasets for cell data
	ncells, npcells = ltype.shape[0], lnods.shape[1]
	lnods, offsets  = _vtkh5_connectivity_and_offsets(lnods)
	ncsize      = ncells*npcells
	ncG, nsG    = int(mpi_reduce(ncells,op='sum',all=True)), int(mpi_reduce(ncsize,op='sum',all=True))
	ncells_dset = file.create_dataset('NumberOfCells',(nparts,),dtype=int)
	nids_dset   = file.create_dataset('NumberOfConnectivityIds',(nparts,),dtype=int)
	conec_dset  = file.create_dataset('Connectivity',(nsG,),dtype=int)
	offst_dset  = file.create_dataset('Offsets',(ncG+nparts,),dtype=int)
	types_dset  = file.create_dataset('Types',(ncG,),dtype=np.uint8)
	# Each partition writes its own part
	# Point data
	istart, iend = ptable.partition_bounds(myrank,points=True)
	npoints_dset[myrank]       = npoints
	points_dset[istart:iend,:] = xyz
	# Cell data
	istart, iend = ptable.partition_bounds(myrank,points=False)
	ncells_dset[myrank] = ncells
	nids_dset[myrank]   = ncsize
	offst_dset[istart+myrank:iend+(myrank+1)] = offsets
	types_dset[istart:iend] = ltype
	# Connectivity
	istart, iend = ptable.partition_bounds(myrank,ndim=npcells,points=False)
	conec_dset[istart:iend] = lnods
	# Return some parameters
	return npG, ncG

@cr('vtkh5IO.save_mesh')
def vtkh5_save_mesh(fname,mesh,ptable,mpio=True):
	'''
	Save the mesh component into a VTKH5 file
	'''
	if mpio and not MPI_SIZE == 1:
		vtkh5_save_mesh_mpio(fname,mesh.xyz,mesh.connectivity,mesh.eltype2VTK,ptable)
	else:
		vtkh5_save_mesh_serial(fname,mesh.xyz,mesh.connectivity,mesh.eltype2VTK)

def vtkh5_save_mesh_serial(fname,xyz,lnods,ltype):
	'''
	Save the mesh component into a VTKH5 file (serial)
	'''
	# Open file for writing
	file = h5py.File(fname,'w')
	# Create the file structure
	main = _vtkh5_create_structure(file)
	# Write the mesh
	_vtkh5_write_mesh_serial(main,xyz,lnods,ltype)
	# Close file
	file.close()

def vtkh5_save_mesh_mpio(fname,xyz,lnods,ltype,ptable):
	'''
	Save the mesh component into a VTKH5 file (serial)
	'''
	# Open file for writing
	file = h5py.File(fname,'w',driver='mpio',comm=MPI_COMM)
	# Create the file structure
	main = _vtkh5_create_structure(file)
	# Write the mesh
	_vtkh5_write_mesh_mpio(main,xyz,lnods,ltype,ptable)
	# Close file
	file.close()


@cr('vtkh5IO.save_field')
def vtkh5_save_field(fname,instant,time,varDict,ptable,mpio=True):
	'''
	Save the mesh component into a VTKH5 file
	'''
	if mpio and not MPI_SIZE == 1:
		vtkh5_save_field_mpio(fname,instant,time,varDict,ptable)
	else:
		vtkh5_save_field_serial(fname,instant,time,varDict)

def vtkh5_save_field_serial(fname,instant,time,varDict):
	'''
	Save the field component into a VTKH5 file (serial)
	'''
	# Open file for writing (append to a mesh)
	file = h5py.File(fname,'a')
	main = file['VTKHDF']
	npoints = int(main['NumberOfPoints'][0])
	# Write dt and instant as field data
	main['FieldData'].create_dataset('InstantValue',(1,),dtype=int,data=instant)
	main['FieldData'].create_dataset('TimeValue',(1,),dtype=float,data=time)
	# Write the variables
	for var in varDict.keys():
		# Obtain in which group to write
		group = 'PointData' if varDict[var].shape[0] == npoints else 'CellData'
		# Create and write
		main[group].create_dataset(var,varDict[var].shape,dtype=varDict[var].dtype,data=varDict[var])
	# Close file
	file.close()

def vtkh5_save_field_mpio(fname,instant,time,varDict,ptable):
	'''
	Save the mesh component into a VTKH5 file (serial)
	'''
	myrank = MPI_RANK
	# Open file for writing
	file = h5py.File(fname,'a',driver='mpio',comm=MPI_COMM)
	main = file['VTKHDF']
	npoints = int(main['NumberOfPoints'][myrank])
	# Write dt and instant as field data
	main['FieldData'].create_dataset('InstantValue',(1,),dtype=int,data=instant)
	main['FieldData'].create_dataset('TimeValue',(1,),dtype=float,data=time)
	# Create the dictionaries
	dsets = {}
	for var in varDict.keys():
		group   = mpi_bcast('PointData' if varDict[var].shape[0] == npoints else 'CellData',root=1)
		npG     = int(mpi_reduce(varDict[var].shape[0],op='sum',all=True)) # Total number of points
		nsizeG  = int(mpi_reduce(varDict[var].shape[1] if len(varDict[var].shape) > 1 else 0,op='max',all=True))
		dsets[var] = main[group].create_dataset(var,(npG,) if nsizeG == 0 else (npG,nsizeG) if nsizeG > 0 else (npG,),dtype=varDict[var].dtype)
	# Write the variables
	for var in varDict.keys():
		istart, iend = ptable.partition_bounds(myrank,points=True if varDict[var].shape[0] == npoints else False)
		if len(varDict[var].shape) == 1: # Scalar field
			dsets[var][istart:iend] = varDict[var]
		else: # Vectorial or tensorial field
			dsets[var][istart:iend,:] = varDict[var]
	# Close file
	file.close()