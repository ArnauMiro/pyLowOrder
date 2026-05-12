#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for RES.
#
# Last rev: 30/04/2026
from __future__ import print_function, division

cimport cython
cimport numpy as np

import numpy as np

#from libc.complex  cimport creal, cimag
cdef extern from "<complex.h>" nogil:
	float  complex I
	# Decomposing complex values
	float cimagf(float complex z)
	float crealf(float complex z)
	double cimag(double complex z)
	double creal(double complex z)
cdef double complex J = 1j
from libc.stdlib     cimport malloc, free
from libc.string     cimport memcpy, memset
from libc.math       cimport sqrt, log, atan2
from ..vmmath.cfuncs cimport real, real_complex
from ..vmmath.cfuncs cimport 
from ..vmmath.cfuncs cimport 
from ..vmmath.cfuncs cimport 
from ..vmmath.cfuncs cimport

from ..utils.cr       import cr, cr_start, cr_stop
from ..utils.errors   import raiseError


## RES run method
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _srun(float[:,:] Phi, float delta, float freq, float f, ):