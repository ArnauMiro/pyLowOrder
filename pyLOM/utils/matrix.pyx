#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for MATRIX.
#
# Last rev: 09/07/2021
from __future__ import print_function, division

cimport cython
cimport numpy as np

import numpy as np

from .cr     import cr_start, cr_stop
from .errors import raiseError

cdef extern from "matrix.h":
	cdef void ctranspose "transpose"(double *A, const int m, const int n, const int bsz)
	cdef double compute_norm(double *A, int start, int n)


def transpose(double[:,:] A, int bsz=0):
	'''
	Transpose a matrix A
	'''
	cr_start('matrix.transpose',0)
	cdef int m = A.shape[0], n = A.shape[1]
	ctranspose(&A[0,0],m,n,bsz)
	cr_stop('matrix.transpose',0)
	return A


def norm(double[:] A, int start=0):
	'''
	Compute the norm of a vector A
	'''
	cr_start('matrix.norm',0)
	cdef int n = A.shape[0]
	cdef double out = 0.
	out = compute_norm(&A[0],start,n)
	cr_stop('matrix.norm',0)
	return out