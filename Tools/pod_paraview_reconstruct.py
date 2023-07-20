#!/bin/env python
'''
This script is intended to be used within POD pyLOM results and a ParaView programmable
filter in ParaView to reconstruct the flow. It consists of two parts, a first script
to set the `Script' view of the programmable filter and a second script to load in the 
`RequestInformation Script' box of the programmable filter.

The reconstruction can be done with any number of modes, as specified by the user.
The output result name is X
'''

## Part 1. Script
PODFILE = '' # Full path to the POD results file saved from pyLOM
ARRNAME = '' # Name of the variable to reconstruct
OUTNAME = 'X' # Output result array
NMODES  = [] # Modes to use during reconstruction (leave empty to use all)

import vtk, h5py, numpy as np
from vtk.util import numpy_support as vtknp

# Helper functions
def load(fname):
	file = h5py.File(fname,'r')
	S = np.array(file['POD']['S'][:])
	V = np.array(file['POD']['V'][:,:])
	# Return
	file.close()
	return S,V

def reconstruct(U,S,V,nmodes,instant):
	if len(nmodes) == 0: nmodes = np.arange(U.shape[1])
	return np.matmul(U[:,nmodes],S[nmodes]*V[nmodes,instant])

# Recover input
pdin       = self.GetInput()
pdout      = self.GetOutput()
unstr_grid = pdin

# Recover instant
outInfo = self.GetOutputInformation(0)
INSTANT = int(outInfo.Get(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())) if outInfo.Has(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()) else 0

# Load U from ParaView
U = vtknp.vtk_to_numpy( unstr_grid.GetPointData().GetArray(ARRNAME) )

# Load S,V from POD file
S,V = load(PODFILE)

# Reconstruct
X = reconstruct(U,S,V,NMODES,INSTANT)

# Store output
vtkarray = vtknp.numpy_to_vtk(X,True,vtk.VTK_DOUBLE)
vtkarray.SetName(OUTNAME)
pdout.GetPointData().AddArray(vtkarray)


## Part 2. RequestInformation Script
DATAFILE = '' # Dataset file where to extract the time vector

import h5py, numpy as np

def load_time(fname):
    file = h5py.File(fname,'r')
    time = np.array(file['time'])
    file.close()
    return time

timeSteps = load_time(DATAFILE)
timeRange = [timeSteps[-1], timeSteps[-1]]

self.GetOutputInformation(0).Set(vtk.vtkStreamingDemandDrivenPipeline.TIME_RANGE(), timeRange, 1)
self.GetOutputInformation(0).Set(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS(), timeSteps, len(timeSteps))