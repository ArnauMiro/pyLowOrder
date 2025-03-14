#!/usr/bin/env python
#
# Example how to perform regression.
#
# Last revision: 10/03/2025
from __future__ import print_function, division

import numpy as np
import pyLOM


# Generate data
x = np.ascontiguousarray(np.array([0, 1, 2, 3],np.double))
y = np.ascontiguousarray(np.array([-1, 0.2, 0.9, 2.1],np.double))

# Rewrite for regression
A = np.ascontiguousarray(np.vstack([x, np.ones(len(x),np.double)]).T)

# PYLOM regression
m,c = pyLOM.math.least_squares(A,y)
pyLOM.pprint(0,'pyLOM LSTSQ:',m,c)
m,c = pyLOM.math.ridge_regresion(A,y,0.01)
pyLOM.pprint(0,'pyLOM RIDGE:',m,c)

# Regression with numpy
m,c = np.linalg.lstsq(A, y)[0]
pyLOM.pprint(0,'numpy LSTSQ:',m,c)

pyLOM.cr_info()