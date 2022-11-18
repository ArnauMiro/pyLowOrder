#!/usr/bin/env python
#
# Example of POD.
#
# Last revision: 19/07/2021
from __future__ import print_function, division

import os, numpy as np
import pyLOM

## Data loading
DATAFILE = 'Examples/Data/CYLINDER.h5'
VARIABLE = 'VELOX'
d        = pyLOM.Dataset.load(DATAFILE)
X        = d[VARIABLE]
t        = d.time
npwin    = 60

pyLOM.cr_start('example',0)

L, P, f = pyLOM.SPOD.run(X, t, nDFT=npwin, nolap=20, remove_mean=True)

pyLOM.SPOD.plotSpectra(f, L)
pyLOM.SPOD.plotMode(P, f, d.xyz, d.mesh, d.info(VARIABLE), f2plot= np.array([1,2,3,4,5,6,7,8]), modes = np.array([1, 2]))

pyLOM.cr_stop('example',0)

pyLOM.cr_info()
pyLOM.show_plots()