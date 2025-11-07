#!/usr/bin/env python
#
# Example on how to run the randomized QR factorization for the GAVI methodology.
#
# Eiximeno, B., A., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025). 
# On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models. 
# Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797
#
# Last revision: 07/11/2025
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import torch
import pyLOM, pyLOM.NN
pyLOM.gpu_device(gpu_per_node=4) # Detect GPU configuration

## Parameters
DATAFILE = '../cylin_velox_train.h5'
VARLIST  = ['velox']

k = 1600 ## Final number of modes we want to retain
o = 100  ## Number of oversampled modes
q = 1    ## Number of power iterations

## Data loading
pyLOM.pprint(0, 'Loading data...', flush=True)
d = pyLOM.Dataset.load(DATAFILE)
X = d['velox']
pyLOM.pprint(0, 'Data loaded!', X.dtype, X.shape, flush=True)

## Run QR
pyLOM.pprint(0, 'Running QR...', flush=True)
Q, B = pyLOM.NN.GAVI.QR(X,k,q=q,osampl=o)
pyLOM.pprint(0, 'QR done!', flush=True)

## Save QR
pyLOM.pprint(0, 'Saving QR...', Q.shape, flush=True)
pyLOM.NN.GAVI.save('QR.h5',Q,B,d.partition_table)
pyLOM.pprint(0, 'QR saved!', flush=True)

pyLOM.cr_info()