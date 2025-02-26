#!/usr/bin/env python
#
# pyLOM, utils.
#
# Parallel routines
#
# Last rev: 14/02/2025
from __future__ import print_function, division

import mpi4py, numpy as np
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import numpy as np

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()

mpi_send = MPI_COMM.send
mpi_recv = MPI_COMM.recv


def worksplit(istart,iend,whoAmI,nWorkers=MPI_SIZE):
	'''
	Divide the work between the processors
	'''
	istart_l, iend_l = istart, iend
	irange = iend - istart
	if (nWorkers < irange):
		# We split normally among processes assuming no remainder
		rangePerProcess = int(np.floor(irange/nWorkers))
		istart_l = istart   + whoAmI*rangePerProcess
		iend_l   = istart_l + rangePerProcess
		# Handle the remainder
		remainder = irange - rangePerProcess*nWorkers
		if remainder > whoAmI:
			istart_l += whoAmI
			iend_l   += whoAmI+1;
		else:
			istart_l += remainder
			iend_l   += remainder
	else:
		# Each process will forcefully conduct one instant.
		istart_l = whoAmI   if whoAmI < iend else iend
		iend_l   = whoAmI+1 if whoAmI < iend else iend

	return istart_l, iend_l


def writesplit(npoints,write_master):
	'''
	Divide the write array between the processors
	'''
	rstart = 1 if not write_master else 0
	istart, iend = 0, 0 
	# Select in which order the processors will write
	if MPI_RANK == rstart:
		# send to next where to start writing
		istart, iend = 0, npoints
		mpi_send(iend,dest=MPI_RANK+1)
	elif MPI_RANK == MPI_SIZE-1:
		# recive from the previous where to start writing
		istart = mpi_recv(source=MPI_RANK-1) 
		iend   = istart + npoints
	else:
		# recive from the previous where to start writing
		istart = mpi_recv(source=MPI_RANK-1) 
		iend   = istart + npoints
		# send to next where to start writing
		mpi_send(iend,dest=MPI_RANK+1) 
	return istart, iend

def split(array,root=0):
	'''
	Split an array among the processors
	'''
	return np.vsplit(array,[worksplit(0,array.shape[0],i)[1] for i in range(MPI_SIZE-1)]) if MPI_RANK==root else None

def is_rank_or_serial(root=0):
	'''
	Return whether the rank is active or True
	in case of a serial run
	'''
	return MPI_RANK == root or MPI_SIZE == 1

def pprint(rank,*args,**kwargs):
	'''
	Print alternative for parallel codes. It works as
	python's print with the rank variable, which can 
	be negative for everyone to print or equal to the
	rank that should print.
	'''
	if MPI_SIZE == 1:
		print(*args,**kwargs)
	elif rank < 0:
		print('Rank %d:'%MPI_RANK,*args,**kwargs)
	elif rank == MPI_RANK:
		print('Rank %d:'%rank,*args,**kwargs)
