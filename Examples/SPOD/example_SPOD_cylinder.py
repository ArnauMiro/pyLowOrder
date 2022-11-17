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

## Data loading
DATAFILE = 'Examples/Data/CYLINDER.h5'
VARIABLE = 'VELOX'
d        = pyLOM.Dataset.load(DATAFILE)
X        = d[VARIABLE]
t        = d.time
npwin    = 60

L, P, f = pyLOM.SPOD.run(X, t, nDFT = npwin, nolap = 40)

for ii in range(L.shape[1]):
    plt.loglog(f, L[:,ii], 'o-')
plt.xlabel('St')
plt.ylabel(r'$\lambda_i$')

#pyLOM.SPOD.plotMode(P, f, d.xyz, d.mesh, d.info(VARIABLE), modes = [1, 2])

plt.show()