#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for SPOD.
#
# Last rev: 02/09/2022
from __future__ import print_function

import numpy as np

from ..vmmath       import vector_norm, vecmat, matmul, temporal_mean, subtract_mean, tsqr_svd
from ..utils.cr     import cr_start, cr_stop
from ..utils.errors import raiseError

## SPOD run method
def run(X, dt, npwin=0, nolap=0, remove_mean=True):
    '''
	Run SPOD analysis of a matrix X.

	Inputs:
	    - X[ndims*nmesh,nt]: data matrix
        - dt:                timestep between adjacent snapshots
        - npwin:             number of points in each window (0 will set default value: ~10% nt)
        - nolap:             number of overlap points between windows (0 will set default value: 50% nwin)
        #- weight:            weight function for each window (1 will set unitary weight for all windows)
        #- window:            give values to the window function
	    - remove_mean:       whether or not to remove the mean flow

	Returns:
	    - L:  modal energy spectra.
	    - P:  SPOD modes, whose spatial dimensions are identical to those of X.
	    - F:  frequency vector.
    ''' 
    cr_start('SPOD.run', 0)

    ## Parse input data to compute the number of blocks (ES POT POSAR EN UNA ALTRA FUNCIÓ)
    M = X.shape[0]
    N = X.shape[1]
    # Number of points per window
    if npwin == 0:
        npwin = int(np.floor(0.1*N))
        npwin = int(2**(np.floor(np.log2(npwin)))) # round-up to 2**n type int
    else:
        npwin = int(npwin)
        npwin = int(2**(np.floor(np.log2(npwin)))) # round-up to 2**n type int
    # Number of overlapping snapshots
    if nolap == 0:
        nolap = int(np.floor(0.5*npwin))
    else:
        if nolap > npwin -1:
            raiseError('Number of overlapping snapshots is too large')
    #Compute number of blocks
    nBlks = int(np.floor((N-nolap)/(npwin-nolap)))



    cr_stop('SPOD.run', 0)
    return nBlks