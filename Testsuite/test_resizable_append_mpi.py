#!/usr/bin/env python
#
# pyLOM Testsuite
# MPI resizable HDF5 append regression test
#
from __future__ import print_function, division

import os
import shutil
import sys
import tempfile
import traceback

import h5py
import numpy as np

MPI = None
if h5py.get_config().mpi:
	try:
		from mpi4py import MPI
	except ImportError:
		pass


NPOINTS    = 6
BLOCK_SIZE = 2
CAPACITY   = 6


def append_block(pyLOM,path,ptable,start):
	'''Append one local, partitioned two-sample block.'''
	rank        = MPI.COMM_WORLD.Get_rank()
	istart,iend = ptable.partition_bounds(rank,points=True)
	order       = np.arange(istart,iend,dtype=np.int32)
	xyz         = np.column_stack((order,10.*order,-order)).astype(np.float64)
	time        = np.arange(start,start+BLOCK_SIZE,dtype=np.float64)
	values      = 100.*order[:,None] + time[None,:]
	dataset     = pyLOM.Dataset(
		xyz=xyz,
		ptable=ptable,
		order=order,
		point=True,
		vars={'time':{'idim':0,'value':time}},
		velocity={'ndim':1,'value':values},
	)
	kwargs = {
		'append':True,
		'append_resizable':True,
		'mode':'a',
		'mpio':True,
	}
	if start == 0: kwargs['append_total_size'] = CAPACITY
	dataset.save(path,**kwargs)


def verify_local(loaded,ptable):
	'''Verify the partition returned by a collective load.'''
	rank        = MPI.COMM_WORLD.Get_rank()
	istart,iend = ptable.partition_bounds(rank,points=True)
	order       = np.arange(istart,iend,dtype=np.int32)
	time        = np.arange(2*BLOCK_SIZE,dtype=np.float64)
	expected_xyz = np.column_stack((order,10.*order,-order)).astype(np.float64)
	np.testing.assert_array_equal(loaded.xyz,expected_xyz)
	np.testing.assert_array_equal(loaded.ordering,order)
	np.testing.assert_array_equal(loaded.get_variable('time'),time)
	np.testing.assert_array_equal(loaded['velocity'],100.*order[:,None] + time[None,:])


def verify_metadata(path):
	'''Verify persistent cursor, reserve, and consolidated arrays on rank zero.'''
	with h5py.File(path,'r') as h5file:
		group       = h5file['DATASET']
		stored_append_type = group.attrs['appendMode']
		if isinstance(stored_append_type,bytes): stored_append_type = stored_append_type.decode()
		assert stored_append_type == 'resizable'
		assert int(group.attrs['appendCursor']) == 2*BLOCK_SIZE
		assert int(group.attrs['appendBlockSize']) == BLOCK_SIZE
		assert not bool(group.attrs['appendNoPartition'])
		assert group.attrs['appendLayoutHash'].shape == (32,)
		assert 'VARIABLES' in group and 'FIELDS' in group
		assert not any(name.startswith('VARIABLES_') for name in group)
		assert not any(name.startswith('FIELDS_') for name in group)
		time     = group['VARIABLES/time/value']
		velocity = group['FIELDS/velocity/value']
		assert time.shape == (CAPACITY,)
		assert time.maxshape == (None,)
		assert velocity.shape == (NPOINTS,CAPACITY)
		assert velocity.maxshape == (NPOINTS,None)
		assert int(group['FIELDS/velocity/vars'][0]) == 2*BLOCK_SIZE
		expected_time = np.arange(2*BLOCK_SIZE,dtype=np.float64)
		rows          = np.arange(NPOINTS,dtype=np.float64)[:,None]
		np.testing.assert_array_equal(time[:2*BLOCK_SIZE],expected_time)
		np.testing.assert_array_equal(
			velocity[:,:2*BLOCK_SIZE],100.*rows + expected_time[None,:]
		)


def main():
	'''Run with ``mpirun -n 2 python Testsuite/test_resizable_append_mpi.py``.'''
	if not h5py.get_config().mpi:
		print('SKIP: h5py must be built with parallel HDF5 support.')
		return 0
	if MPI is None:
		print('SKIP: mpi4py is required for the MPI resizable append test.')
		return 0
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	if comm.Get_size() != 2:
		if rank == 0: print('SKIP: run this test with exactly two MPI ranks.')
		return 0
	import pyLOM

	tmpdir = tempfile.mkdtemp(prefix='pylom-resizable-mpi-') if rank == 0 else None
	tmpdir = comm.bcast(tmpdir,root=0)
	path   = os.path.join(tmpdir,'resizable_append.h5')
	ptable = pyLOM.PartitionTable.new(2,NPOINTS,NPOINTS)

	append_block(pyLOM,path,ptable,0)
	append_block(pyLOM,path,ptable,BLOCK_SIZE)
	loaded = pyLOM.Dataset.load(path,ptable=ptable,mpio=True)

	local_error = None
	try:
		verify_local(loaded,ptable)
	except Exception:
		local_error = 'rank %d local verification:\n%s' % (rank,traceback.format_exc())
	errors = comm.gather(local_error,root=0)
	if rank == 0:
		try:
			verify_metadata(path)
		except Exception:
			errors.append('rank 0 metadata verification:\n%s' % traceback.format_exc())
		failure = '\n'.join(error for error in errors if not error is None)
	else:
		failure = None
	failure = comm.bcast(failure,root=0)
	comm.Barrier()
	if rank == 0: shutil.rmtree(tmpdir)
	if failure: raise AssertionError(failure)
	if rank == 0: print('PASS: MPI resizable append')
	return 0


if __name__ == '__main__':
	sys.exit(main())
