#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Utils - Error handling routines.
#
# Last rev: 09/07/2021
from __future__ import print_function, division

import sys, mpi4py, numpy as np
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI


mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


def raiseError(errmsg):
	'''
	Raise a controlled error and abort execution on
	all processes.
	'''
	print('%d - %s' % (mpi_rank,errmsg),file=sys.stderr,flush=True)
	mpi_comm.Abort(1)


def raiseWarning(warnmsg,allranks=False):
	'''
	Raise a controlled warning but don't abort execution on
	all processes.
	'''
	if allranks:
		pprint(-1,'Warning! %d - %s' % (mpi_rank,warnmsg),file=sys.stderr,flush=True)
	else:
		pprint(0,'Warning! %s' % (warnmsg),file=sys.stderr,flush=True)