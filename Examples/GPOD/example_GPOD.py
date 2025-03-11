#!/usr/bin/env python
#
# Example of GPOD - Reconstruct the Gappy Snapshot Matrix.
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
xyzc = m.xyzc


## Define percentage for missing data
missing_percentage = 0.15  # 15% missing data
# Generate random mask
np.random.seed(5)
random_mask = np.random.choice(
    [0, 1], size=X.shape, p=[missing_percentage, 1 - missing_percentage]
)
# Create incomplete snapshot matrix
incomplete_snapshot = X * random_mask

# Database reconstruction
num_iter = 50

# GappyPOD model setup
gappy_model_recons = pyLOM.GPOD.GappyPOD(
    centered=False,
    apply_truncation=True,
    truncation_param=-0.98,
    reconstruction_method="ridge",
    ridge_lambda=0.1,
)


## Reconstruct database
X_recons, eig_spec_iter, c_e = gappy_model_recons.reconstruct_full_set(
    incomplete_snapshot,
    num_iter,
)


## Compute metrics
mae_recons  = pyLOM.math.MAE(X, X_recons)
rmse_recons = pyLOM.math.RMSE(X, X_recons)
r2_recons   = pyLOM.math.r2(X, X_recons)
print(f"MAE_database = {mae_recons}\nRMSE_database = {rmse_recons}\nR2_database = {r2_recons}")


## Eigenvalue spectrum and cumulative energy for original data
PSI, S, V = pyLOM.POD.run(X, remove_mean=True)
c_e_exact = np.cumsum(S) / np.sum(S)
eig_spec_exact = S**2 / np.sum(S**2)
mode_index = np.arange(1, len(c_e_exact) + 1)


## Plot Eigenvalue spectrum
fig, ax = plt.subplots(figsize=(8, 9))
ax.scatter(mode_index[:-1], eig_spec_exact[:-1], c="black", label="Exact")
for it, label in zip(
    [0, 10, 25, num_iter - 1], ["1 iter", "10 iter", "25 iter", f"{num_iter} iter"]
):
    ax.plot(mode_index[:-1], eig_spec_iter[:, it][:-1], label=label)
ax.set_yscale("log")
ax.set_title("Ridge Gappy POD")
ax.legend()
ax.set_xlabel("Eigenvalue")
ax.set_ylabel("Eigenvalue spectrum")


## Plot Cumulative energy
fig, ax = plt.subplots(figsize=(8, 9))
ax.scatter(mode_index[:-1], c_e_exact[:-1], c="black", label="Exact")
for it, label in zip(
    [0, 10, 25, num_iter - 1], ["1 iter", "10 iter", "25 iter", f"{num_iter} iter"]
):
    ax.plot(mode_index[:-1], c_e[:, it][:-1], label=label)
ax.axhline(y=0.9, color="black", linestyle="--", label="90% Energy")
ax.set_title("Ridge Gappy POD")
ax.legend()
ax.set(xlabel="Eigenvalue", ylabel="Cumulative energy")


## Dump to ParaView
d.add_field('VELXR',1,X_recons)
pyLOM.io.pv_writer(m,d,'flow',basedir='out/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=['VELOX','VELXR'],fmt='vtkh5')
pyLOM.POD.plotSnapshot(m,d,vars=['VELXR'],instant=0,component=0,cmap='jet',cpos='xy')


pyLOM.cr_info()
plt.show()