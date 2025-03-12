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
# Reconstruct Gappy Vector
##########################################
# Select snapshot and create gappy vector
snap = PARAMS['snap']
np.random.seed(5)
snapshot_POD = np.delete(X, snap, axis=1)  # Remove snapshot from training data
velox_gappy = pyLOM.GPOD.utils.set_random_elements_to_zero(X[:, snap], 99.9)

# Perform Gappy POD
gappy_model = pyLOM.GPOD.GappyPOD(**PARAMS['gpod'])
gappy_model.fit(snapshot_POD)  # Fit model
velox_recons = gappy_model.predict(velox_gappy)  # Reconstruct gappy vector

# Compute and display metrics
mae_vector  = pyLOM.math.MAE(np.ascontiguousarray(X[:,snap]), velox_recons)
rmse_vector = pyLOM.math.RMSE(np.ascontiguousarray(X[:,snap]), velox_recons, relative=False)
r2_vector   = pyLOM.math.r2(np.ascontiguousarray(X[:,snap]), velox_recons)


## Testsuite output
pyLOM.pprint(0,'TSUITE VELOX       =',velox_recons.min(),velox_recons.max(),velox_recons.mean())
pyLOM.pprint(0,'TSUITE MAE  VECTOR = %e'%mae_vector)
pyLOM.pprint(0,'TSUITE RMSE VECTOR = %e'%rmse_vector)
pyLOM.pprint(0,'TSUITE R2   VECTOR = %e'%r2_vector)


## Show and print timings
pyLOM.cr_info()
pyLOM.pprint(0,'End of output')