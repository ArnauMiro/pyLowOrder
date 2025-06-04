#!/bin/env python
#
# Conversion to NLR7301 dataset to 
# pyLOM v3.0 format
#
# 27/09/2024
from __future__ import print_function, division

import os,glob,numpy as np, netCDF4 as NC4
import pyLOM


## Datasets
DATASETS = ['test','train','val']
NPOINTS  = 597 # According to documentation


## Create a serial partition table
ptable = pyLOM.PartitionTable.new(1,NPOINTS,NPOINTS)
print(ptable)


## Load point normals
normals = np.load("normals.npz")["normals"]


## Loop on the datasets
for dset in DATASETS:
    print(dset)
    # Generate a list of the available files
    filelist = glob.glob(os.path.join(dset,'*'))
    # Obtain available Mach and AoA
    case   = [int(f.split('_')[1][4:]) for f in filelist]
    Mvec   = [float(f.split('_')[2][1:]) for f in filelist]
    AoAvec = [float(f.split('_')[3][3:]) for f in filelist]
    # Create input vectors
    xyz = np.zeros((NPOINTS,2),np.double)
    X   = np.zeros((NPOINTS,len(Mvec)),np.double) # assuming len(Mvec) == len(AoAvec) according to documentation
    # Read dataset and populate
    ii = 0
    for c,M,AoA in zip(case,Mvec,AoAvec):
        ncfile = NC4.Dataset(os.path.join(dset,'Snap_Case%04d_M%.5f_AoA%.5f'%(c,M,AoA)))
        xyz[:,0] = ncfile.variables['x'][:597]
        xyz[:,1] = ncfile.variables['z'][:597]
        X[:,ii]  = ncfile.variables['cp'][:597]
        ncfile.close()
        ii += 1
    # Create a pyLOM dataset
    d = pyLOM.Dataset(xyz=xyz, ptable=ptable, order=np.arange(xyz.shape[0]), point=True,
        # Add the variables
        vars  = {
            'Mach':{'idim':0,'value':np.array(Mvec)},
            'AoA' :{'idim':0,'value':np.array(AoAvec)},
        },
        # Now add all the arrays to be stored in the dataset
        # It is important to convert them as C contiguous arrays
        CP      = {'ndim':1,'value':X},
        NORMALS = {'ndim':2,'value':np.expand_dims(normals.flatten(),axis=1)},
    )
    print(d)
    d.save(f'{dset.upper()}.h5') # Store dataset