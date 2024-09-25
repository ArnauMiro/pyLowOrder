#!/usr/bin/env python
#
# PYLOM Testsuite
# Build channel dataset for the testsuite
#
# Last revision: 03/08/2021
from __future__ import print_function, division

import numpy as np
import pyAlya, pyLOM


BASEDIR        = '../'
DSETDIR        = './'
CASESTR        = 'channel'
VARLIST        = ['VELOC','PRESS']
START, DT, END = 500000, 500, 505000

# In case of restart, load the previous data
listOfInstants = [ii for ii in range(START,END,DT)]
ni = len(listOfInstants)


## Load pyAlya mesh
mesh = pyAlya.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=False,read_massm=False)
pyAlya.pprint(0,'Run (%d instants)...' % len(listOfInstants),flush=True)


## Create POD dataset
p = pyLOM.PartitionTable.from_pyQvarsi(mesh.partition_table,has_master=True)
m = pyLOM.Mesh.from_pyQvarsi(mesh,ptable=p)


## Build dataset from the instants
time    = np.zeros((len(listOfInstants),),np.double)
X_PRESS = np.zeros((mesh.nnod,ni),dtype=np.double) # POD matrix, VELOC and PRESS
X_VELOX = np.zeros((mesh.nnod,ni),dtype=np.double) # POD matrix, VELOC and PRESS
X_VELOY = np.zeros((mesh.nnod,ni),dtype=np.double) # POD matrix, VELOC and PRESS
X_VELOZ = np.zeros((mesh.nnod,ni),dtype=np.double) # POD matrix, VELOC and PRESS
for ii,instant in enumerate(listOfInstants):
	if ii%100 == 0: pyAlya.pprint(1,'Processing instant %d...'%instant,flush=True)
	# Read data
	field, header = pyAlya.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR)
	# Store time
	time[ii] = header.time
	# Store the POD matrix
	X_PRESS[:,ii] = field['PRESS']
	X_VELOX[:,ii] = field['VELOC'][:,0]
	X_VELOY[:,ii] = field['VELOC'][:,1]
	X_VELOZ[:,ii] = field['VELOC'][:,2]


## Create dataset for pyLOM
d = pyLOM.Dataset(xyz=m.xyz, ptable=p, order=m.pointOrder, point=True,
	# Add the time as the only variable
	vars  = {'time':{'idim':0,'value':time}},
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	PRESS = {'ndim':1,'value':X_PRESS},
	VELOX = {'ndim':1,'value':X_VELOX},
	VELOY = {'ndim':1,'value':X_VELOY},
	VELOZ = {'ndim':1,'value':X_VELOZ},
)


## Store dataset
# Nopartition will eliminate the partition of alya and allow the dataset to be run
# with any number of processors but will increase the save time to the disk
# Use with great care!!
m.save('%s.h5'%CASESTR.upper(),nopartition=True) 
d.save('%s.h5'%CASESTR.upper(),nopartition=True) 


pyAlya.cr_info()
pyLOM.cr_info()