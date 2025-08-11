#!/usr/bin/env python
#
# PYLOM Testsuite POD
#
# Last revision: 20/09/2024
from __future__ import print_function, division

import sys, os, json, numpy as np
import pyLOM


## Parameters
DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
PARAMS    = json.loads(str(sys.argv[4]).replace("'",'"').lower())

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d.X(*VARIABLES)
t = d.get_variable('time')

## Run POD
comp,scores = pyLOM.PCA.run(X,**PARAMS['run']) # PSI are POD modes
# Truncate according to a residual
print("KKK",comp.shape)

## Testsuite output
pyLOM.pprint(0,'TSUITE components  =',comp.min(),comp.max(),comp.mean())
pyLOM.pprint(0,'TSUITE scores      =',scores.min(),scores.max(),scores.mean())

## Show and print timings
pyLOM.cr_info()
pyLOM.pprint(0,'End of output')