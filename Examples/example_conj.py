#!/usr/bin/env python
#
# Example how to perform the conjugate.
#
# Last revision: 01/02/2023
from __future__ import print_function, division

import numpy as np
import pyLOM

V  = np.array([
	[5.+3.*1j, 8.+5.*1j, 9.-2.*1j, 12.+4.*1j],
	[5.+3.*1j, 8.+5.*1j, 9.-2.*1j, 12.+4.*1j]
])
print('Original: ', V)

Vconj = pyLOM.math.conj(V)
print('Conjugate: ', Vconj)

pyLOM.cr_info()
