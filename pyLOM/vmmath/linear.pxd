#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Linear operator module - exporting of Cython functions.
#
# Last rev: 07/09/2026

cimport numpy as np

# Float precision 
cdef tuple _slinear_operator(float[:,:] Y, float[:,:] Z, float r)
cdef tuple _sconcatenate(list X, int remove_mean)
cdef tuple _sseparate(float[:,:] X, int remove_mean)

# Double precision
cdef tuple _dlinear_operator(double[:,:] Y, double[:,:] Z, double r)
cdef tuple _dconcatenate(list X, int remove_mean)
cdef tuple _dseparate(double[:,:] X, int remove_mean)

# Complex single precision
cdef tuple _cresolvent(float[:,:] A, np.complex64_t f)

# Complex double precision
cdef tuple _zresolvent(double[:,:] A, np.complex128_t f)