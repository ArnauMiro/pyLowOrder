#!/bin/env python
#
# Extract DATASET from Ensight
#
# Last rev: 16/01/2024
from __future__ import print_function, division

import numpy as np
import pyLOM

CASENAME = 'AoA_0p00_AoS_0p00_FL310/FMTC_C_M0p56_AoA_0p00_AoS_0p00_FL310_solution.surface.pval.14310.case'

# Read CASE file from Ensight Gold
case  = pyLOM.io.Ensight_readCase2(CASENAME)
ids   = case.get_geometry_model().get_part_ids()
names = case.get_geometry_model().get_part_names()

# Create a dataset per each of the parts
for partID,partName in zip(ids,names):
	name = ''.join(e for e in partName if e.isalnum()) # Delete special characters
	print(name)
	# Read the geometry file
	xyz, conec, eltype = pyLOM.io.Ensight_readGeo2(case.get_geometry_model(),partID)
	cellOrder  = np.arange(conec.shape[0])
	pointOrder = np.arange(xyz.shape[0])
	# Create the mesh
	mesh = pyLOM.Mesh('UNSTRUCT',xyz,conec,eltype,cellOrder,pointOrder)
	print(mesh)
	# Create a serial partition table
	ptable = pyLOM.PartitionTable.new(1,mesh.ncells,mesh.npoints)
	mesh.partition_table = ptable
	print(ptable)
	# Create a pyLOM dataset
	d = pyLOM.Dataset(xyz=mesh.xyz, ptable=ptable, order=mesh.pointOrder, point=True,
		# Add the time as the only variable
		vars  = {'time':{'idim':0,'value':np.zeros((1,), dtype=np.double)}},
	)
	# Add variables to dataset
	X_VAR = np.zeros((mesh.npoints,1),dtype=np.double)
	for v in case.get_variables():
		X_VAR[:,0] = pyLOM.io.Ensight_readField2(case.get_variable(v),partID)
		d.add_variable(v,True,1,X_VAR.copy())
	print(d)
	# Store dataset to disk - pyLOM format
	mesh.save(name+'.h5',mode='w')  
	d.save(name+'.h5')
	# Store dataset to disk - vtkhdf format
	pyLOM.io.pv_writer(mesh,d,name+'.hdf',vars=case.get_variables(),fmt='vtkhdf')

pyLOM.cr_info()