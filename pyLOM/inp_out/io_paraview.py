#!/usr/bin/env python
#
# pyLOM, IO
#
# Output for ParaView readable formats
#
# Last rev: 18/09/2024

import os, numpy as np

from .io_vtkh5      import vtkh5_save_mesh, vtkh5_link_mesh, vtkh5_save_field
from ..utils.cr     import cr
from ..utils.errors import raiseError


@cr('Writer.write')
def pv_writer(Mesh,Dataset,casestr,basedir='./',idim=0,instants=[0],times=[0.],vars=[],fmt='vtkh5',mode='w'):
	'''
	Store the data using various formats for ParaView.

	This method differs from save in the fact that save is used 
	to recover the field, write only outputs the data.
	'''
	os.makedirs(basedir,exist_ok=True)
	if fmt.lower() in ['vtkh5','vtkhdf']:
		VTKHDF5Writer(Mesh,Dataset,casestr,basedir,instants,times,vars,idim,mode)
	else:
		raiseError('Format <%s> not implemented!'%fmt)


def VTKHDF5Writer(mesh,dset,casestr,basedir,instants,times,varnames,idim,mode):
	'''
	VTKHDF dataset writer
	'''
	# Create a mesh file
	meshname = os.path.join(basedir,'%s-mesh-vtk.hdf'%(casestr))
	# We set the mode at vtkh5_save_mesh as it will create the file
	vtkh5_save_mesh(meshname,mesh,mesh.partition_table,mode=mode)
	# Loop the instants
	for instant, time in zip(instants,times):
		fieldname = os.path.join(basedir,'%s-%08d-vtk.hdf'%(casestr,instant))
		# Link the mesh on the file
		# We set the mode at vtkh5_link_mesh as it will create the file
		vtkh5_link_mesh(fieldname,'./%s-mesh-vtk.hdf'%(casestr),mode=mode)
		# Write the data on the file
		varDict = {}
		for v in varnames:
			sliced     = tuple([np.s_[:]] + [0 if i != idim else instant for i in range(len(dset[v].shape)-1)])
			varDict[v] = mesh.reshape_var(dset[v][sliced],dset.info(v))
		# Save field must append to the existing file
		vtkh5_save_field(fieldname,instant,time,dset.point,varDict,mesh.partition_table,mode='a')
