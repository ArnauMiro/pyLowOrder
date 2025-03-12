#!/usr/bin/env python
#
# PYLOM Testsuite GPOD.
#
# Last revision: 26/02/2025
from __future__ import print_function, division
import mpi4py
import pyLOM.GPOD

mpi4py.rc.recv_mprobe = False

import sys, json, numpy as np
import pyLOM

## Parameters
DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
PARAMS    = json.loads(str(sys.argv[4]).replace("'",'"').lower())

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE, ptable=m.partition_table)
X = d.X(*VARIABLES)
t = d.get_variable('time')


##########################################
# Reconstruct the Gappy Snapshot Matrix
##########################################
# Generate random mask
np.random.seed(5)
random_mask = np.random.choice(
    [0, 1], size=X.shape, p=[PARAMS['missing_percentage'], 1 - PARAMS['missing_percentage']]
)
# Create incomplete snapshot matrix
incomplete_snapshot = X * random_mask


# Database reconstruction
num_iter = 20

# GappyPOD model setup
gappy_model_recons = pyLOM.GPOD.GappyPOD(**PARAMS['gpod'])

# Reconstruct database
X_recons, eig_spec_iter, c_e = gappy_model_recons.reconstruct_full_set(
    incomplete_snapshot,
    PARAMS['num_iter'],
)

# Compute metrics
mae_recons  = pyLOM.math.MAE(X, X_recons)
rmse_recons = pyLOM.math.RMSE(X, X_recons, relative=False)
r2_recons   = pyLOM.math.r2(X, X_recons)


## Testsuite output
pyLOM.pprint(0,'TSUITE VELOX =',X_recons.min(),X_recons.max(),X_recons.mean())
pyLOM.pprint(0,'TSUITE MAE   = %e'%mae_recons)
pyLOM.pprint(0,'TSUITE RMSE  = %e'%rmse_recons)
pyLOM.pprint(0,'TSUITE R2    = %e'%r2_recons)


## Show and print timings
pyLOM.cr_info()
pyLOM.pprint(0,'End of output')
