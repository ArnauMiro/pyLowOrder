#!/bin/env python
#
# Extract DATASET for POD using pyAlya
#
# Last rev: 28/10/2021
from __future__ import print_function, division

import os, numpy as np
import pyAlya, pyLOM


BASEDIR        = '../sod2d_gitlab/'
DSETDIR        = './'
CASESTR        = 'cube'
VARLIST        = ['u_x']
START, DT, END = 1, 200, 2001

# In case of restart, load the previous data
listOfInstants = [ii for ii in range(START,END,DT)]
ni = len(listOfInstants)

## Load pyAlya mesh
mesh = pyAlya.Mesh.read(CASESTR,basedir=BASEDIR,fmt='sod',read_codno=False,read_commu=False,read_massm=False)
pyAlya.pprint(0,'Run (%d instants)...' % len(listOfInstants),flush=True)

## Create POD dataset
m = pyLOM.Mesh.from_pyAlya(mesh)
p = pyLOM.PartitionTable.from_pyAlya(mesh.partition_table,has_master=False)
d = pyLOM.Dataset(ptable=p, mesh=m, time=np.zeros((ni,),dtype=np.double))

## Build dataset from the instants
X_VELOX = np.zeros((mesh.nnod,ni),dtype=np.double) # POD matrix, VELOC and PRESS
for ii,instant in enumerate(listOfInstants):
	if ii%100 == 0: pyAlya.pprint(1,'Processing instant %d...'%instant,flush=True)
	# Read data
	field, header = pyAlya.Field.read(CASESTR, VARLIST, instant, mesh.xyz, basedir=BASEDIR, fmt='sod')
	# Store time
	d.time[ii] = header.time
	# Store the POD matrix
	X_VELOX[:,ii] = field['u_x']


## Add variables to the dataset
d.add_variable('VELOX',True,1,X_VELOX)


## Store dataset
# Nopartition will eliminate the partition of alya and allow the dataset to be run
# with any number of processors but will increase the save time to the disk
# Use with great care!!
d.save('%s.h5'%CASESTR,nopartition=True) 

pyAlya.cr_info()
pyLOM.cr_info()
