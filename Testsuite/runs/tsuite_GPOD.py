#!/usr/bin/env python
#
# PYLOM Testsuite GPOD.
#
# Last revision: 26/02/2025
from __future__ import print_function, division
import mpi4py
import pyLOM.GPOD

mpi4py.rc.recv_mprobe = False

import sys, os, json, numpy as np
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
xyzc = m.xyzc

 ##########################################
# Reconstruct Gappy Vector
##########################################
# Select snapshot and create gappy vector
snap = 15
np.random.seed(5)
snapshot_POD = np.delete(X, snap, axis=1)  # Remove snapshot from training data
velox_gappy = pyLOM.GPOD.utils.set_random_elements_to_zero(X[:, snap], 99.9)

# Perform Gappy POD
gappy_model = pyLOM.GPOD.GappyPOD(
    centered=False,
    apply_truncation=True,
    truncation_param=-0.99,
    reconstruction_method="ridge",
    ridge_lambda=0.01,
)
gappy_model.fit(snapshot_POD)  # Fit model
velox_recons = gappy_model.predict(velox_gappy)  # Reconstruct gappy vector

# Compute and display metrics
rmse_vector = pyLOM.math.RMSE(X[:, snap], velox_recons)

##########################################
# Reconstruct the Gappy Snapshot Matrix
##########################################
# Define percentage for missing data
missing_percentage = 0.15  # 15% missing data
# Generate random mask
np.random.seed(5)
random_mask = np.random.choice(
    [0, 1], size=X.shape, p=[missing_percentage, 1 - missing_percentage]
)
# Create incomplete snapshot matrix
incomplete_snapshot = X * random_mask


# Database reconstruction
num_iter = 20

# GappyPOD model setup
gappy_model_recons = pyLOM.GPOD.GappyPOD(
    centered=False,
    apply_truncation=True,
    truncation_param=-0.98,
    reconstruction_method="ridge",
    ridge_lambda=0.1,
)

# Reconstruct database
X_recons, eig_spec_iter, c_e = gappy_model_recons.reconstruct_full_set(
    incomplete_snapshot,
    num_iter,
)

# Compute metrics
rmse_recons = pyLOM.math.RMSE(X, X_recons)


## Testsuite output
pyLOM.pprint(0,'TSUITE RMSE VECTOR  = %e'%rmse_vector)
pyLOM.pprint(0,'TSUITE RMSE RECONS  = %e'%rmse_recons)
pyLOM.pprint(0,'TSUITE VELOX     =',velox_recons.min(),velox_recons.max(),velox_recons.mean())
pyLOM.pprint(0,'TSUITE X     =',X_recons.min(),X_recons.max(),X_recons.mean())

## Show and print timings
pyLOM.cr_info()
pyLOM.pprint(0,'End of output')
