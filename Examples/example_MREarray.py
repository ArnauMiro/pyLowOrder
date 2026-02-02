#!/usr/bin/env python
#
# Example running MREarray operator.
#
# Last revision: 02/02/2026
from __future__ import print_function, division

import numpy as np
import pyLOM

def MRE_array(A,B,axis1):
    num  = (A-B)
    numg = np.sum(num*num,axis=axis1)
    deng = np.sum(B*B,axis=axis1)
    return numg/deng


## Generate data
m = 12                 # number of rows
n = 283                # number of columns (must be > m)
dtype = np.double      # np.float32 or np.double
noise_scale = 1.0      # controls magnitude of randomVal
random_seed = 42       # for reproducibility (optional)


## Generate data
rng = np.random.default_rng(random_seed)
A   = rng.random((m, n), dtype=dtype)
randomVal = noise_scale * rng.random((m, n), dtype=dtype)
B   = A + 0.5 * randomVal


## Perform operation
# Numpy
out = MRE_array(A,B,1)
pyLOM.pprint(0,out.shape,out,flush=True)
# pyLOM
out = pyLOM.math.MRE_array(A,B,axis=1)
pyLOM.pprint(0,out.shape,out,flush=True)


pyLOM.cr_info()