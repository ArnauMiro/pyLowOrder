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

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()

MPI_RDONLY = MPI.MODE_RDONLY
MPI_WRONLY = MPI.MODE_WRONLY
MPI_CREATE = MPI.MODE_CREATE


# Expose functions from MPI library
mpi_create_op = MPI.Op.Create
mpi_wtime     = MPI.Wtime
mpi_file_open = MPI.File.Open

mpi_nanmin = mpi_create_op(lambda v1,v2,dtype : np.nanmin([v1,v2]),commute=True)
mpi_nanmax = mpi_create_op(lambda v1,v2,dtype : np.nanmax([v1,v2]),commute=True)
mpi_nansum = mpi_create_op(lambda v1,v2,dtype : np.nansum([v1,v2]),commute=True)


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


def mpi_barrier():
	'''
	Implements the barrier
	'''
	MPI_COMM.Barrier()


def mpi_send(f,dest,tag=0):
	'''
	Implements the send operation
	'''
	MPI_COMM.send(f,dest,tag=tag)


def mpi_recv(**kwargs):
	'''
	Implements the recieve operation
	'''
	return MPI_COMM.recv(**kwargs)


def mpi_sendrecv(buff,**kwargs):
	'''
	Implements the sendrecv operation
	'''
	return MPI_COMM.sendrecv(buff,**kwargs)


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
		if not isinstance(sendbuff,np.ndarray) and not isinstance(sendbuff,list): sendbuff = [sendbuff]
		if all:
			out = np.array(MPI_COMM.allgather(sendbuff))
			return np.concatenate(out,axis=0)
		else:
			out = np.array(MPI_COMM.gather(sendbuff,root=root))
			return np.concatenate(out,axis=0) if MPI_RANK == root else None
	return sendbuff


def mpi_reduce(sendbuff,root=0,op='sum',all=False):
	'''
	Reduce an array from all the processors.
	'''
	if MPI_SIZE > 1:
		if isinstance(op,str):
			if 'sum'    in op: opf = MPI.SUM
			if 'max'    in op: opf = MPI.MAX
			if 'min'    in op: opf = MPI.MIN
			if 'nanmin' in op: opf = mpi_nanmin
			if 'nanmax' in op: opf = mpi_nanmax
			if 'nansum' in op: opf = mpi_nansum
		else:
			opf = op
		if all:
			return MPI_COMM.allreduce(sendbuff,op=opf)
		else:
			out = MPI_COMM.reduce(sendbuff,op=opf,root=root)
			return out if root == MPI_RANK else sendbuff
	else:
		return sendbuff


def mpi_bcast(sendbuff,root=0):
	'''
	Implements the broadcast operation
	'''
	return MPI_COMM.bcast(sendbuff,root=root)


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