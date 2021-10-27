#!/usr/bin/env python
#
# pyLOM, utils.
#
# Parallel routines
#
# Last rev: 25/10/2021
from __future__ import print_function, division

import sys, mpi4py, numpy as np
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

# MPI basics
MPI_COMM = MPI.COMM_WORLD      # Communications macro
MPI_RANK = MPI_COMM.Get_rank() # Who are you? who? who?
MPI_SIZE = MPI_COMM.Get_size() # Total number of processors used (workers)


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


def split(array,root=0):
	'''
	Split an array among the processors
	'''
	return np.vsplit(array,MPI_SIZE) if MPI_RANK==root else None


def is_rank_or_serial(root=0):
	'''
	Return whether the rank is active or True
	in case of a serial run
	'''
	return MPI_RANK == root or MPI_SIZE == 1


def mpi_scatter(sendbuff,root=0,do_split=False):
	'''
	Send an array among the processors and split
	if necessary.
	'''
	if MPI_SIZE > 1:
		return MPI_COMM.scatter(split(sendbuff,root=root),root=root) if do_split else MPI_COMM.scatter(sendbuff,root=root)
	return sendbuff


def mpi_gather(sendbuff,root=0,all=False):
	'''
	Gather an array from all the processors.
	'''
	if MPI_SIZE > 1:
		if all:
			return np.vstack(MPI_COMM.allgather(sendbuff))
		else:
			out = MPI_COMM.gather(sendbuff,root=root)
			return np.vstack(out) if MPI_RANK == root else None
	return sendbuff


def mpi_reduce(sendbuff,root=0,op='sum'):
	if MPI_SIZE > 1:
		if isinstance(op,str):
			if 'sum' in op: opf = MPI.SUM
			if 'max' in op: opf = MPI.MAX
			if 'min' in op: opf = MPI.MIN
		else:
			opf = op
		out = MPI_COMM.reduce(sendbuff,op=opf,root=root)
		return out if root == MPI_RANK else sendbuff
	else:
		return sendbuff


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