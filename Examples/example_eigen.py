#!/usr/bin/env python
#
# Example of eigenvalues so that it returns
# the same output as MATLAB.
#
# Last revision: 30/09/2021
from __future__ import print_function, division

import numpy as np
import pyLOM


## Define matrix A 4x4
A = np.array([
	[1.0000,0.5000,0.3333,0.2500],
    [0.5000,1.0000,0.6667,0.5000],
    [0.3333,0.6667,1.0000,0.7500],
    [0.2500,0.5000,0.7500,1.0000]],dtype=np.double,order='C')


## Run eigenvalues from numpy
pyLOM.cr_start('eigen numpy',0)
w,v = np.linalg.eig(A)
print('Numpy:')
print(w.shape,w)
print(v.shape,v)
pyLOM.cr_stop('eigen numpy',0)


## Run eigenvalyes from DMD
pyLOM.cr_start('eigen cython',0)
delta,w,v = pyLOM.DMD.eigen(A)
print('DMD:')
print(delta.shape,delta)
print(w.shape,w)
print(v.shape,v)
pyLOM.cr_stop('eigen cython',0)


pyLOM.cr_info()