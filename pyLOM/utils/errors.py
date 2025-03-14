#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Utils - Error handling routines.
#
# Last rev: 09/07/2021
from __future__ import print_function, division

import sys, numpy as np
from .mpi import MPI_RANK, MPI_COMM


def round(value,precision):
	'''
	Truncate array by a certain precision
	'''
	fact  = 10**precision
	return np.round(value*fact)/fact


def raiseError(errmsg):
	'''
	Raise a controlled error and abort execution on
	all processes.
	'''
	print('%d - %s' % (MPI_RANK,errmsg),file=sys.stderr,flush=True)
	MPI_COMM.Abort(1)


def raiseWarning(warnmsg,allranks=False):
	'''
	Raise a controlled warning but don't abort execution on
	all processes.
	'''
	if allranks:
		print('Warning! %d - %s' % (MPI_RANK,warnmsg),file=sys.stderr,flush=True)
	else:
		if MPI_RANK == 0: print(0,'Warning! %s' % (warnmsg),file=sys.stderr,flush=True)