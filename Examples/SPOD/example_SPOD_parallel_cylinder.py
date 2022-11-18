#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np
import pyLOM
import matplotlib.pyplot as plt
import scipy
from pyLOM.utils.parall import mpi_gather

## Data loading
DATAFILE = 'Examples/Data/CYLINDER.h5'
VARIABLE = 'VELOX'
d        = pyLOM.Dataset.load(DATAFILE)
X        = d[VARIABLE]
t        = d.time
npwin    = 60

pyLOM.cr_start('example',0)

L,Pi,f = pyLOM.SPOD.run(X, t, nDFT=npwin, nolap=20, remove_mean=True)

Mi     = X.shape[0] 
nBlks  = L.shape[1]
Xg     = mpi_gather(X,root=0)
xyz    = mpi_gather(d.xyz,root=0)

if pyLOM.is_rank_or_serial(0):
    M  = Xg.shape[0]
    P  = np.zeros((nBlks*M, f.shape[0]))

for iBlk in range(nBlks):
    Pib = mpi_gather(Pi[iBlk*Mi:(iBlk+1)*Mi,:], root=0)
    if pyLOM.is_rank_or_serial(0):
        P[iBlk*M:(iBlk+1)*M, :] = Pib

if pyLOM.is_rank_or_serial(0):
    pyLOM.SPOD.plotSpectra(f, L)
    pyLOM.SPOD.plotMode(P, f, xyz, d.mesh, d.info(VARIABLE), f2plot= np.array([1,2,3,4,5,6,7,8]), modes = np.array([1, 2]))
    pyLOM.show_plots()

pyLOM.cr_stop('example',0)
pyLOM.cr_info()