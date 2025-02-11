#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 07/02/2025
from __future__ import print_function, division
import mpi4py
import pyLOM.GPOD

mpi4py.rc.recv_mprobe = False

import os, numpy as np
import pyLOM
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

## Parameters
DATAFILE = "./Testsuite/DATA/CYLINDER.h5"
VARIABLE = "VELOX"


## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE, ptable=m.partition_table)
X = d[VARIABLE]
t = d.get_variable('time')
xyzc = m.xyzc
# breakpoint()

 ##########################################
# Reconstruct Gappy Vector
##########################################
# Select snapshot and create gappy vector
snap = 15
snapshot_POD = np.delete(X, snap, axis=1)  # Remove snapshot from training data
velox_gappy = pyLOM.GPOD.utils.set_random_elements_to_zero(X[:, snap], 99.9)

# Perform Gappy POD
gappy_model = pyLOM.GPOD.GappyPOD(
    centered=False,
    apply_truncation=True,
    truncation_method="energy",
    truncation_param=0.99,
    reconstruction_method="ridge",
    ridge_lambda=0.01,
)
gappy_model.fit(snapshot_POD)  # Fit model
velox_recons = gappy_model.predict(velox_gappy)  # Reconstruct gappy vector

# Compute and display metrics
mae = mean_absolute_error(X[:, snap], velox_recons)
rmse = root_mean_squared_error(X[:, snap], velox_recons)
r2 = r2_score(X[:, snap], velox_recons)
print(f"MAE_snapshot = {mae}\nRMSE_snapshot = {rmse}\nR2_snapshot = {r2}")


# Plot Gappy Vector
loc_gappy = np.where(velox_gappy != 0)[0]
xyzx_gappy = xyzc[loc_gappy]
plt.scatter(
    xyzx_gappy[:, 0], xyzx_gappy[:, 1], c=velox_gappy[loc_gappy], cmap="jet", s=1
)
plt.title("Gappy Vector")
plt.colorbar(label="Velox")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Plot Reconstructed Vector
plt.scatter(xyzc[:, 0], xyzc[:, 1], c=velox_recons, cmap="jet", s=1)
plt.title("Reconstructed Vector")
plt.colorbar(label="Velox")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Absolute Error Plot
error_abs = np.abs(X[:, snap] - velox_recons)
plt.scatter(xyzc[:, 0], xyzc[:, 1], c=error_abs, cmap="coolwarm", s=5)
plt.colorbar(label="Error")
plt.title("Absolute Error")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

##########################################
# Reconstruct the Gappy Snapshot Matrix
##########################################
# Define percentage for missing data
missing_percentage = 0.20  # 20% missing data
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
    truncation_method="energy",
    truncation_param=0.99,
    reconstruction_method="ridge",
    ridge_lambda=0.1,
)

# Reconstruct database
snapshot_matrix_recons, eig_spec_iter, c_e = gappy_model_recons.reconstruct_full_set(
    incomplete_snapshot,
    num_iter,
)

# Compute metrics
mae = mean_absolute_error(X, snapshot_matrix_recons)
rmse = root_mean_squared_error(X, snapshot_matrix_recons)
r2 = r2_score(X, snapshot_matrix_recons)
print(f"MAE_database = {mae}\nRMSE_database = {rmse}\nR2_database = {r2}")

# Eigenvalue spectrum and cumulative energy for original data
PSI, S, V = pyLOM.POD.run(X, remove_mean=True)
c_e_exact = np.cumsum(S) / np.sum(S)
eig_spec_exact = S**2 / np.sum(S**2)
mode_index = np.arange(1, len(c_e_exact) + 1)

# Plot Eigenvalue spectrum
fig, ax = plt.subplots(figsize=(8, 9))
ax.scatter(mode_index[:-1], eig_spec_exact[:-1], c="black", label="Exact")
for it, label in zip(
    [0, 10, 25, num_iter - 1], ["1 iter", "10 iter", "25 iter", "50 iter"]
):
    ax.plot(mode_index[:-1], eig_spec_iter[:, it][:-1], label=label)
ax.set_yscale("log")
ax.set_title("Ridge Gappy POD")
ax.legend()
ax.set_xlabel("Eigenvalue")
ax.set_ylabel("Eigenvalue spectrum")
plt.show()

# Plot Cumulative energy
fig, ax = plt.subplots(figsize=(8, 9))
ax.scatter(mode_index[:-1], c_e_exact[:-1], c="black", label="Exact")
for it, label in zip(
    [0, 10, 25, num_iter - 1], ["1 iter", "10 iter", "25 iter", "50 iter"]
):
    ax.plot(mode_index[:-1], c_e[:, it][:-1], label=label)
ax.axhline(y=0.9, color="black", linestyle="--", label="90% Energy")
ax.set_title("Ridge Gappy POD")
ax.legend()
ax.set(xlabel="Eigenvalue", ylabel="Cumulative energy")
plt.show()
