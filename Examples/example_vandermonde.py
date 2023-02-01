
#!/usr/bin/env python
#
# Example how to compute the Vandermonde matrix.
#
# Last revision: 01/02/2023
from __future__ import print_function, division

import numpy as np
import pyLOM

real  = np.array([5., 8., 9., 12.])
imag  = np.array([4., 1., 7., -5.])
ncols = 3

Vand = pyLOM.math.vandermonde(real, imag, real.shape[0], ncols)
print('Vandermonde matrix: ', Vand)

real = np.array([5., 8., 9., 12.])
imag = np.array([4., 1., 7., -5.])
time = np.arange(0, 5, dtype = np.double)

Vand = pyLOM.math.vandermondeTime(real, imag, real.shape[0], time)
print('Vandermonde matrix: ', Vand)

pyLOM.cr_info()