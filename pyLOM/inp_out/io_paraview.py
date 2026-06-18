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


VTKHDF_MESH_FMT         = '%s-mesh.vtkhdf'
VTKHDF_FIELD_FMT        = '%s-%08d.vtkhdf'
VTKHDF_MESH_FMT_LEGACY  = '%s-mesh-vtk.hdf'
VTKHDF_FIELD_FMT_LEGACY = '%s-%08d-vtk.hdf'


@cr('Writer.write')
def pv_writer(Mesh,Dataset,casestr,basedir='./',idim=0,instants=[0],times=[0.],vars=[],fmt='vtkh5',mode='w',legacy=True):
	'''
	Store the data using various formats for ParaView.

	This method differs from save in the fact that save is used 
	to recover the field, write only outputs the data.

	Args:
		Mesh (Mesh).
		Dataset (Datset).
		casestr (str): Name of the case
		basedir (str, optional): Base directory. Default is ``'./'``.
		idim (int, optional): Dimension of the variable to save. Default is 0.
		instants (list[int], optional): List of instants of time to save.
			Usually of the form ``[i for i in range(n)]``. Default is ``[0]``.
		times (list[float], optional): List of physical time corresponding to
			each of the ``instants``. Default is ``[0.]``.
		vars (list[str], optional): List of the names of the variables to write.
			Default is the empty list.
		fmt (str, optional): Format in which to save the mesh. Default is
			``'vtkh5'``, which is the only supported. Note that ``'vtkhdf'`` is
			equivalent to ``'vtkh5'``.
		mode (str, optional) : ``'w'`` (default) for overwrite and
			``'a'`` for append.
		legacy (bool, optional) : Legacy mode. Default is ``True``.
	'''
	os.makedirs(basedir,exist_ok=True)
	if fmt.lower() in ['vtkh5','vtkhdf']:
		VTKHDF5Writer(Mesh,Dataset,casestr,basedir,instants,times,vars,idim,mode,legacy)
	else:
		raiseError('Format <%s> not implemented!'%fmt)


def VTKHDF5Writer(mesh,dset,casestr,basedir,instants,times,varnames,idim,mode,legacy):
	'''
	VTKHDF dataset writer.

	Args:
		mesh (Mesh).
		dset (Datset).
		casestr (str): Name of the case
		basedir (str): Base directory.
		instants (list[int]): List of instants of time to save.
			Usually of the form ``[i for i in range(n)]``.
		times (list[float]): List of physical time corresponding to
			each of the ``instants``.
		varnames (list[str]): List of the names of the variables to write.
		idim (int): Dimension of the variable to save.
		mode (str) : ``'w'`` for overwrite and
			``'a'`` for append.
		legacy (bool) : Legacy mode.
	'''
	vtkh5_mesh_fmt  = VTKHDF_MESH_FMT_LEGACY  if legacy else VTKHDF_MESH_FMT
	vtkh5_field_fmt = VTKHDF_FIELD_FMT_LEGACY if legacy else VTKHDF_FIELD_FMT
	# Create a mesh file
	meshname = os.path.join(basedir,vtkh5_mesh_fmt%(casestr))
	# We set the mode at vtkh5_save_mesh as it will create the file
	vtkh5_save_mesh(meshname,mesh,mesh.partition_table,mode=mode)
	# Loop the instants
	for instant, time in zip(instants,times):
		fieldname = os.path.join(basedir,vtkh5_field_fmt%(casestr,instant))
		# Link the mesh on the file
		# We set the mode at vtkh5_link_mesh as it will create the file
		vtkh5_link_mesh(fieldname,'./%s'%(vtkh5_mesh_fmt%casestr),mode=mode)
		# Write the data on the file
		varDict = {}
		for v in varnames:
			sliced     = tuple([np.s_[:]] + [0 if i != idim else instant for i in range(len(dset[v].shape)-1)])
			varDict[v] = mesh.reshape_var(dset[v][sliced],dset.info(v))
		# Save field must append to the existing file
		vtkh5_save_field(fieldname,instant,time,dset.point,varDict,mesh.partition_table,mode='a')
