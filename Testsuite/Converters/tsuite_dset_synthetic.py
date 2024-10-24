#!/usr/bin/env python
#
# PYLOM Testsuite
# Build synthetic dataset for the testsuite
#
# Last revision: 24/10/2024
from __future__ import print_function, division

import numpy as np
from scipy.io import loadmat

import pyLOM


## Parameters
OUTFILE     = './DATA/SYNTHETIC.h5'
n_samples   = 1250
noise_level = 0.05


## Generate input features
xyz  = np.random.randn(n_samples,3).astype(np.float32)
data = np.zeros((n_samples,1),np.float32)
xyz[:,2]   = 0.
data[:,0]  = np.sin(xyz[:, 0]) + np.cos(xyz[:, 1]) + 0.5 * xyz[:, 0] ** 2
# Add noise
data[:,0] += noise_level*np.random.randn(n_samples)


## Build time instants
time = np.array([1],np.float32)


## Create partition table and order
ptable = pyLOM.PartitionTable.new(1,n_samples,n_samples)
order  = np.arange(n_samples,dtype=np.int32)
print(ptable)


## Create dataset for pyLOM
# Select a limited number of time instants
d = pyLOM.Dataset(xyz=xyz, ptable=ptable, order=order, point=True,
	# Add the time as the only variable
	vars = {'time':{'idim':0,'value':time}},
	# Now add all the arrays to be stored in the dataset
	# It is important to convert them as C contiguous arrays
	DATA = {'ndim':1,'value':data},
)
print(d)
d.save(OUTFILE) # Store dataset


pyLOM.cr_info()