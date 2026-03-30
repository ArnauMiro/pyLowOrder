#!/usr/bin/env python
#
# Example how to perform tsqr decomposition.
#
# Last revision: 01/02/2023
from __future__ import print_function, division

import numpy as np
import pyLOM

## Parameters
DATAFILE = './DATA/CYLINDER.h5'
VARIABLE = 'VELOC'

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d[VARIABLE]

Q, R     = pyLOM.math.tsqr(X)
Xr       = pyLOM.math.matmul(Q, R)
Ek       = pyLOM.math.energy(Xr,X)
pyLOM.pprint(0, 'Partial energy recovered', Ek, flush=True)

pyLOM.cr_info()