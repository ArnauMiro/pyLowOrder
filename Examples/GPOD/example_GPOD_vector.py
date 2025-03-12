#!/usr/bin/env python
#
# Example of GPOD - Reconstruct Gappy Vector.
#
# Last revision: 26/02/2025
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np
import matplotlib.pyplot as plt
import pyLOM


## Parameters
DATAFILE = "./DATA/CYLINDER.h5"
VARIABLE = "VELOX"


## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE, ptable=m.partition_table)
X = d[VARIABLE]
t = d.get_variable('time')


## Select snapshot and create gappy vector
snap = 15
np.random.seed(5)
snapshot_POD = np.delete(X, snap, axis=1)  # Remove snapshot from training data
velox_gappy = pyLOM.GPOD.utils.set_random_elements_to_zero(X[:, snap], 99.9)


## Perform Gappy POD
gappy_model = pyLOM.GPOD.GappyPOD(
    centered=False,
    apply_truncation=True,
    truncation_param=-0.99,
    reconstruction_method="ridge",
    ridge_lambda=0.01,
)
gappy_model.fit(snapshot_POD)  # Fit model
velox_recons = gappy_model.predict(velox_gappy)  # Reconstruct gappy vector


## Compute and display metrics
mae_vector  = pyLOM.math.MAE(np.ascontiguousarray(X[:,snap]), velox_recons)
rmse_vector = pyLOM.math.RMSE(np.ascontiguousarray(X[:,snap]), velox_recons)
r2_vector   = pyLOM.math.r2(np.ascontiguousarray(X[:,snap]), velox_recons)
print(f"MAE_snapshot = {mae_vector}\nRMSE_snapshot = {rmse_vector}\nR2_snapshot = {r2_vector}")


# Plot Reconstructed Vector
plt.figure(figsize=(8,6),dpi=100)
plt.scatter(m.xyzc[:, 0], m.xyzc[:, 1], c=velox_recons, cmap="jet", s=1)
plt.title("Reconstructed Vector")
plt.colorbar(label="Velox")
plt.xlabel("X")
plt.ylabel("Y")


## Dump to ParaView
d.add_field('VELXR',1,velox_recons)
d.add_field('ERABS',1,np.abs(X[:, snap] - velox_recons))
pyLOM.io.pv_writer(m,d,'flow',basedir='out/flow',instants=[snap],times=[t[snap]],vars=['VELOX','VELXR','ERABS'],fmt='vtkh5')
pyLOM.POD.plotSnapshot(m,d,vars=['VELXR'],instant=0,component=0,cmap='jet',cpos='xy')
pyLOM.POD.plotSnapshot(m,d,vars=['ERABS'],instant=0,component=0,cmap='coolwarm',cpos='xy')


pyLOM.cr_info()
plt.show()