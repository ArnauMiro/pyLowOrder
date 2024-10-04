#!/usr/bin/env python
#
# pyLOM, IO
#
# Output for ParaView readable formats
#
# Last rev: 18/09/2024

import os, numpy as np

from .io_vtkh5      import vtkh5_save_mesh, vtkh5_link_mesh, vtkh5_save_field
from .io_ensight    import Ensight_writeGeo, Ensight_writeField
from ..utils.cr     import cr
from ..utils.errors import raiseError


@cr('Writer.write')
def pv_writer(Mesh,Dataset,casestr,basedir='./',idim=0,instants=[0],times=[0.],vars=[],fmt='vtkh5'):
	'''
	Store the data using various formats for ParaView.

	This method differs from save in the fact that save is used 
	to recover the field, write only outputs the data.
	'''
	os.makedirs(basedir,exist_ok=True)
	if fmt.lower() in ['vtk']:
		raiseError('VTK format not implemented! Use vtkhdf instead')
	elif fmt.lower() in ['ensi','ensight']:
		EnsightWriter(Mesh,Dataset,casestr,basedir,instants,vars,idim)
	elif fmt.lower() in ['vtkh5','vtkhdf']:
		VTKHDF5Writer(Mesh,Dataset,casestr,basedir,instants,times,vars,idim)
	else:
		raiseError('Format <%s> not implemented!'%fmt)


def VTKHDF5Writer(mesh,dset,casestr,basedir,instants,times,varnames,idim):
	'''
	Ensight dataset writer
	'''
	# Create a mesh file
	meshname = os.path.join(basedir,'%s-mesh-vtk.hdf'%(casestr))
	vtkh5_save_mesh(meshname,mesh,mesh.partition_table)
	# Loop the instants
	for instant, time in zip(instants,times):
		fieldname = os.path.join(basedir,'%s-%08d-vtk.hdf'%(casestr,instant))
		# Link the mesh on the file
		vtkh5_link_mesh(fieldname,'./%s-mesh-vtk.hdf'%(casestr))
		# Write the data on the file
		varDict = {}
		for v in varnames:
			sliced     = tuple([np.s_[:]] + [0 if i != idim else instant for i in range(len(dset[v].shape)-1)])
			varDict[v] = mesh.reshape_var(dset[v][sliced],dset.info(v))
		vtkh5_save_field(fieldname,instant,time,dset.point,varDict,mesh.partition_table)

def EnsightWriter(mesh,dset,casestr,basedir,instants,varnames,idim):
	'''
	Ensight dataset writer
	'''
	# Create the filename for the geometry
	geofile = os.path.join(basedir,'%s.ensi.geo'%casestr)
	header = {
		'descr'  : 'File created with pyAlya tool\nmesh file',
		'nodeID' : 'assign',
		'elemID' : 'assign',
		'partID' : 1,
		'partNM' : 'Volume Mesh',
		'eltype' : mesh.eltype2ENSI
	}
	# Write geometry file
	Ensight_writeGeo(geofile,mesh.xyz,mesh.connectivity+1,header) # Python index start at 0
	# Write instantaneous fields
	binfile_fmt = '%s.ensi.%s-%06d'
	# Define Ensight header
	header = {
		'descr'  : 'File created with pyLOM',
		'partID' : 1,
		'partNM' : 'part',
		'eltype' : mesh.eltype2ENSI
	}
	# Loop the selected instants
	for var in varnames:
		# Recover variable information
		info  = dset.info(var)
		field = dset[var]
		# Variable has temporal evolution
		header['eltype'] = mesh.eltype2ENSI
		# Loop requested instants
		for instant in instants:
			filename = os.path.join(basedir,binfile_fmt % (casestr,var,instant+1))
			# Reshape variable for Ensight file
			sliced = tuple([np.s_[:]] + [0 if i != idim else instant for i in range(len(dset[v].shape)-1)])
			f = mesh.reshape_var(dset[v][sliced],dset.info(v))
			Ensight_writeField(filename,f,header)