#!/usr/bin/env python
#
# PYLOM Testsuite Isomap
#
# Last revision: 11/01/2025
from __future__ import print_function, division

import sys
import pyLOM

## Parameters
DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d.X(*VARIABLES)
t = d.get_variable('time')


## Run MDS
Y = pyLOM.MANIFOLD.mds(X,2)
## Testsuite output
values = ''
for val in Y[0,:]:
    values += str(abs(val)) + ' '
pyLOM.pprint(0,'TSUITE X = ',values)
values = ''
for val in Y[1,:]:
    values += str(abs(val)) + ' '
pyLOM.pprint(0,'TSUITE Y = ',values)

## Show and print timings
pyLOM.cr_info()
pyLOM.pprint(0,'End of output')