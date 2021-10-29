#!/bin/env python
#
# Extract DATASET for POD using pyAlya
#
# Last rev: 28/10/2021
from __future__ import print_function, division

import os, numpy as np
import pyAlya, pyLOM


BASEDIR        = '../'
CASESTR        = 'chan'
VARLIST        = ['VELOC','PRESS']
START, DT, END = 1,1,415+1

# In case of restart, load the previous data
listOfInstants = [ii for ii in range(START,END,DT)]
ni = len(listOfInstants)


## Create the subdomain mesh
mesh = pyAlya.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=False,read_massm=False)
pyAlya.pprint(0,'Run (%d instants)...' % len(listOfInstants),flush=True)


## Create POD dataset
#m = {'type':'struct3D','nx':65,'ny':65,'nz':66}
m = {'type':'unstructured','elkind':'HEX08','nnod':mesh.nnodG,'nel':mesh.nelG}
d = pyLOM.Dataset(mesh=m, xyz=mesh.xyz, time=np.zeros((ni,),dtype=np.double), 
		pointOrder=mesh.lninv, 
		cellOrder=np.arange(mesh.nel,dtype=np.int32)
)


## Loop time instants
X_PRESS = np.zeros((mesh.nnod,ni),dtype=np.double)   # POD matrix, VELOC and PRESS
X_VELOC = np.zeros((3*mesh.nnod,ni),dtype=np.double) # POD matrix, VELOC and PRESS
for ii,instant in enumerate(listOfInstants):
	# Read data
	field, header = pyAlya.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR)
	# Store time
	d.time[ii] = header.time
	# Store the POD matrix
	X_PRESS[:,ii] = field['PRESS']
	X_VELOC[:,ii] = field['VELOC'].reshape((3*mesh.nnod,),order='C')


## Add variables to the dataset
d.add_variable('PRESS',True,1,ni,X_PRESS)
d.add_variable('VELOC',True,3,ni,X_VELOC)


## Store dataset
d.save('%s.h5'%CASESTR,write_master=False) # Now there are two, a master and an apprentice

pyAlya.cr_info()
pyLOM.cr_info()
