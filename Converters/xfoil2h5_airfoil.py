#!/usr/bin/env python
#
# Conversor of TXT file to HDF5 file.
#
# Last revision: 25/10/2024
from __future__ import print_function, division

import numpy as np
import pyLOM


## Parameters
TXTFILE = './DATA/dataset_xfoil.txt'
OUTFILE = './DATA/AIRFOIL.h5'


## Read dataset
data = np.loadtxt(
    TXTFILE, 
    usecols=range(0,5), 
    delimiter="\t", 
    comments="#", 
    dtype=np.float32,
    skiprows=1
)
print(data.shape)

# Obtain unique Reynolds and AoA
Re_u  = np.unique(data[:, 0])
AoA_u = np.unique(data[:, 1])
print('Reynolds (unique):', len(Re_u), Re_u)
print('AoA (unique):', len(AoA_u), AoA_u)

# Obtain Reynolds and AoA
AoA, Re = [], []
for RR in Re_u:
    # Find a subdataset for the given Re
    d1 = data[data[:,0] == RR]
    for AA in AoA_u:
        # Find a subdataset for the given AoA
        d2 = d1[d1[:,1] == AA]
        if len(d2) > 0: 
            Re.append(RR)
            AoA.append(AA)
Re, AoA  = np.array(Re), np.array(AoA)
print('Reynolds:', len(Re))
print('AoA:', len(AoA))

# Obtain the number of points on the airfoil
npoints = 99 # number of points in the airfoil
xy = data[:npoints,2:4]
print('xy: ',xy.shape)

# Obtain the CP
cp = np.ascontiguousarray(data[:,4:5].reshape((npoints,len(AoA)),order='F'))
print('cp: ',cp.shape)

# Create a serial partition table
ptable = pyLOM.PartitionTable.new(1,npoints,npoints)
print(ptable)

# Create a pyLOM dataset
d = pyLOM.Dataset(xyz=xy, ptable=ptable, order=np.arange(xy.shape[0]), point=True,
    # Add the variables
    vars  = {
        'Re'  : {'idim':0,'value':Re},
        'AoA' : {'idim':0,'value':AoA},
    },
    # Now add all the arrays to be stored in the dataset
    # It is important to convert them as C contiguous arrays
    cp      = {'ndim':1,'value':cp},
)
print(d)
d.save(OUTFILE) # Store dataset