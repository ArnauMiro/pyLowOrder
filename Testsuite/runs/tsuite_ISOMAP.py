#!/usr/bin/env python
#
# PYLOM Testsuite Isomap
#
# Last revision: 11/01/2025
from __future__ import print_function, division

import sys, json
import pyLOM

## Parameters
DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
PARAMS    = json.loads(str(sys.argv[4]).replace("'",'"'))

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d.X(*VARIABLES)
t = d.get_variable('time')

# Define the number of neighbours
K = PARAMS['K']

## Run Isomap
Y,R,_ = pyLOM.MANIFOLD.isomap(X,2,K)
## Testsuite output
pyLOM.pprint(0,'TSUITE R = %.10f'%R)
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