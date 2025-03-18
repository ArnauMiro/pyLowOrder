#!/usr/bin/env python
#
# pyLOM, utils.
#
# Parallel MPI routines
#
# Last rev: 14/02/2025
from __future__ import print_function, division

import mpi4py, numpy as np
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from .parall import split
from .nvtxp  import nvtxp

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


def mpi_barrier():
	'''
	Implements the barrier
	'''
	MPI_COMM.Barrier()


@nvtxp('mpi_send',color='red')
def mpi_send(f,dest,tag=0):
	'''
	Implements the send operation
	'''
	MPI_COMM.send(f,dest,tag=tag)


@nvtxp('mpi_recv',color='red')
def mpi_recv(**kwargs):
	'''
	Implements the recieve operation
	'''
	return MPI_COMM.recv(**kwargs)


@nvtxp('mpi_sendrecv',color='red')
def mpi_sendrecv(buff,**kwargs):
	'''
	Implements the sendrecv operation
	'''
	return MPI_COMM.sendrecv(buff,**kwargs)


@nvtxp('mpi_scatter',color='red')
def mpi_scatter(sendbuff,root=0,do_split=False):
	'''
	Send an array among the processors and split
	if necessary.
	'''
	if MPI_SIZE > 1:
		return MPI_COMM.scatter(split(sendbuff,root=root),root=root) if do_split else MPI_COMM.scatter(sendbuff,root=root)
	return sendbuff


@nvtxp('mpi_gather',color='red')
def mpi_gather(sendbuff,root=0,all=False):
	'''
	Gather an array from all the processors.
	'''
	if MPI_SIZE > 1:
		if not isinstance(sendbuff,np.ndarray) and not isinstance(sendbuff,list): sendbuff = [sendbuff]
		if all:
			out = MPI_COMM.allgather(sendbuff)
			return np.concatenate(out,axis=0)
		else:
			out = MPI_COMM.gather(sendbuff,root=root)
			return np.concatenate(out,axis=0) if MPI_RANK == root else None
	return sendbuff


@nvtxp('mpi_reduce',color='red')
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


@nvtxp('mpi_bcast',color='red')
def mpi_bcast(sendbuff,root=0):
	'''
	Implements the broadcast operation
	'''
	return MPI_COMM.bcast(sendbuff,root=root)
