#!/bin/env python
#
# Extract DATASET for POD using pyQvarsi
# Original dataset computed with MIGALE
#
# Last rev: 23/07/2025

import numpy as np
import pyQvarsi, pyLOM


## Parameters (for now)
DATADIR   = 'sphere'
CASE      = 'af5b'
VARIABLES = ['density','pressure','temperature','velocity']
INSTANTS  = np.arange(1713,1791+1)


## Read mesh
pyQvarsi.pprint(0,'Reading mesh data...',end=' ',flush=True)
mesh = pyQvarsi.MeshMIGALE.read(CASE,basedir=DATADIR)
pyQvarsi.pprint(0,'DONE!',flush=True)

# Create pyLOM mesh
pyQvarsi.pprint(0,'Create pyLOM mesh...',end=' ',flush=True)
p = pyLOM.PartitionTable.from_pyQvarsi(mesh.partition_table,ndime=mesh.ndim,has_master=False)
m = pyLOM.Mesh.from_pyQvarsi(mesh,ptable=p)
pyQvarsi.pprint(0,'DONE!',flush=True)


## Build dataset from the instants
ni      = INSTANTS.shape[0]
time    = np.zeros((ni,),np.double)
X_DENSI = np.zeros((mesh.nnod,ni),dtype=np.double)
X_PRESS = np.zeros((mesh.nnod,ni),dtype=np.double)
X_TEMPE = np.zeros((mesh.nnod,ni),dtype=np.double)
X_VELOX = np.zeros((mesh.nnod,ni),dtype=np.double)
X_VELOY = np.zeros((mesh.nnod,ni),dtype=np.double)
X_VELOZ = np.zeros((mesh.nnod,ni),dtype=np.double)

# Read instants
for ii,instant in enumerate(INSTANTS):
    if ii%10 == 0: pyQvarsi.pprint(0,'Processing instant %d...'%instant,flush=True)
    # Read MIGALE field
    field = pyQvarsi.FieldMIGALE.read(CASE,VARIABLES,instant,mesh.xyz,basedir=DATADIR,ptable=mesh.partition_table)
    # Load variables to pyLOM
    X_DENSI[:,ii] = field['density']
    X_PRESS[:,ii] = field['pressure']
    X_TEMPE[:,ii] = field['temperature']
    X_VELOX[:,ii] = field['velocity'][:,0]
    X_VELOY[:,ii] = field['velocity'][:,1]
    X_VELOZ[:,ii] = field['velocity'][:,2]

# Create dataset for pyLOM
d = pyLOM.Dataset(xyz=m.xyz, ptable=p, order=m.pointOrder, point=True,
    # Add the time as the only variable
    vars  = {'time':{'idim':0,'value':time}},
    # Now add all the arrays to be stored in the dataset
    # It is important to convert them as C contiguous arrays
    DENSI = {'ndim':1,'value':X_DENSI},
    PRESS = {'ndim':1,'value':X_PRESS},
    TEMPE = {'ndim':1,'value':X_TEMPE},
    VELOX = {'ndim':1,'value':X_VELOX},
    VELOY = {'ndim':1,'value':X_VELOY},
    VELOZ = {'ndim':1,'value':X_VELOZ},
)


## Store dataset
# Nopartition will eliminate the partition of alya and allow the dataset to be run
# with any number of processors but will increase the save time to the disk
# Use with great care!!
m.save('migsphere.h5',nopartition=True,mode='w') # This will overwrite any existing file
d.save('migsphere.h5',nopartition=True) # This is already on append mode


pyQvarsi.cr_info()
pyLOM.cr_info()
