#!/usr/bin/env python
#
# Example of GPR.
#
# Last revision: 19/02/2025
from __future__ import print_function, division

import sys, os, json, numpy as np
import pyLOM, pyLOM.GPR

## Parameters
DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
PARAMS    = json.loads(str(sys.argv[4]).replace("'",'"'))

## Data loading
d = pyLOM.Dataset.load(DATAFILE)

# Data selection
selected_snapshot = 5
pressure_snapshot = d.X(*VARIABLES)[:, selected_snapshot]
coordinates = d.xyz

np.random.seed(42)
masked_pressure = pressure_snapshot*(np.random.rand(len(pressure_snapshot)) < 0.65)
nonzero_indices = np.nonzero(masked_pressure)[0]
filtered_coordinates = coordinates[nonzero_indices]

x_train = filtered_coordinates
y_train = masked_pressure[nonzero_indices]
x_pred = coordinates

# GPR
gpr = pyLOM.GPR.SF_GPR(input_dim=2)
print(gpr.kernel.available_kernels)  # Dict of available kernels
print(
    gpr.kernel.get_kernel_parameters("RBF")
)  # Dict of available parameters for the selecter kernel
kernel = gpr.kernel.Matern32(variance=0.1, lengthscale=1.0, ARD=True)
gpr.fit(x_train, y_train, kernel)
gpr.display_model()
result = gpr.predict(x_pred)
mean = result["mean"]
std = result["std"]

rmse = pyLOM.math.RMSE(result["mean"].flatten(),pressure_snapshot)

## Testsuite output
pyLOM.pprint(0,'TSUITE RMSE      = %e'%rmse)
pyLOM.pprint(0,'TSUITE GPR MEAN  =',mean.min(),mean.max(),mean.mean())
pyLOM.pprint(0,'TSUITE GPR STD   =',std.min(),std.max(),std.mean())

## Show and print timings
pyLOM.cr_info()
pyLOM.pprint(0,'End of output')