#!/usr/bin/env python
#
# pyLOM Testsuite
# Resizable HDF5 append regression tests
#
from __future__ import print_function, division

import os
import importlib
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import pyLOM
import pyLOM.inp_out.io_h5 as h5io

dataset_module = importlib.import_module('pyLOM.dataset')


class ResizableAppendTests(unittest.TestCase):
	NPOINTS = 5

	def setUp(self):
		self.tmpdir = tempfile.TemporaryDirectory()
		self.xyz = np.arange(3*self.NPOINTS,dtype=np.float64).reshape((self.NPOINTS,3))
		self.order = np.arange(self.NPOINTS,dtype=np.int32)
		self.ptable = pyLOM.PartitionTable.new(1,self.NPOINTS,self.NPOINTS)

	def tearDown(self):
		self.tmpdir.cleanup()

	def _path(self,name):
		return os.path.join(self.tmpdir.name,name)

	def _dataset(self,start,ninstants=2):
		time = np.arange(start,start+ninstants,dtype=np.float64)
		point = np.arange(self.NPOINTS,dtype=np.float64)[:,None]
		return pyLOM.Dataset(
			xyz=self.xyz,
			ptable=self.ptable,
			order=self.order,
			point=True,
			vars={
				'time':{'idim':0,'value':time},
				'step':{'idim':0,'value':time+1000.},
			},
			velocity={'ndim':1,'value':100.*point + time[None,:]},
			pressure={'ndim':1,'value':-10.*point - time[None,:]},
		)

	def _save_append(self,path,start,ninstants=2,total_size=None):
		kwargs = {
			'append':True,
			'append_resizable':True,
			'mode':'a',
			'mpio':False,
		}
		if not total_size is None: kwargs['append_total_size'] = total_size
		self._dataset(start,ninstants).save(path,**kwargs)

	def _assert_values(self,dataset,start,ninstants):
		time = np.arange(start,start+ninstants,dtype=np.float64)
		point = np.arange(self.NPOINTS,dtype=np.float64)[:,None]
		np.testing.assert_array_equal(dataset.xyz,self.xyz)
		np.testing.assert_array_equal(dataset.ordering,self.order)
		np.testing.assert_array_equal(dataset.get_variable('time'),time)
		np.testing.assert_array_equal(dataset.get_variable('step'),time+1000.)
		np.testing.assert_array_equal(dataset['velocity'],100.*point + time[None,:])
		np.testing.assert_array_equal(dataset['pressure'],-10.*point - time[None,:])

	def test_resizable_append_uses_consolidated_groups(self):
		path = self._path('consolidated.h5')
		self._save_append(path,0)

		with h5py.File(path,'r') as h5file:
			group = h5file['DATASET']
			stored_append_type = group.attrs['appendMode']
			if isinstance(stored_append_type,bytes): stored_append_type = stored_append_type.decode()
			self.assertIn('VARIABLES',group)
			self.assertIn('FIELDS',group)
			self.assertFalse(any(name.startswith('VARIABLES_') for name in group))
			self.assertFalse(any(name.startswith('FIELDS_') for name in group))
			self.assertEqual(stored_append_type,'resizable')
			self.assertEqual(int(group.attrs['appendCursor']),2)
			self.assertEqual(int(group.attrs['appendBlockSize']),2)
			self.assertFalse(bool(group.attrs['appendNoPartition']))
			self.assertEqual(group.attrs['appendLayoutHash'].shape,(32,))
			self.assertEqual(group['VARIABLES/time/value'].maxshape,(None,))
			self.assertEqual(group['FIELDS/velocity/value'].maxshape,(self.NPOINTS,None))
			self.assertIsNotNone(group['VARIABLES/time/value'].chunks)
			self.assertIsNotNone(group['FIELDS/velocity/value'].chunks)

	def test_reserved_capacity_is_trimmed_to_written_values(self):
		path = self._path('reserved.h5')
		self._save_append(path,0,total_size=6)

		with h5py.File(path,'r') as h5file:
			group = h5file['DATASET']
			self.assertEqual(group['VARIABLES/time/value'].shape,(6,))
			self.assertEqual(group['FIELDS/velocity/value'].shape,(self.NPOINTS,6))
			self.assertEqual(int(group.attrs['appendCursor']),2)
			self.assertEqual(int(group['FIELDS/velocity/vars'][0]),2)

		loaded = pyLOM.Dataset.load(path,mpio=False)
		self._assert_values(loaded,0,2)
		self.assertEqual(loaded['velocity'].shape,(self.NPOINTS,2))

		self._save_append(path,2)
		with h5py.File(path,'r') as h5file:
			group = h5file['DATASET']
			self.assertEqual(group['VARIABLES/time/value'].shape,(6,))
			self.assertEqual(group['FIELDS/velocity/value'].shape,(self.NPOINTS,6))
			self.assertEqual(int(group.attrs['appendCursor']),4)
			self.assertEqual(int(group['FIELDS/velocity/vars'][0]),4)

		loaded = pyLOM.Dataset.load(path,mpio=False)
		self._assert_values(loaded,0,4)

	def test_append_grows_beyond_reserved_capacity(self):
		path = self._path('growth.h5')
		self._save_append(path,0,total_size=4)
		self._save_append(path,2)
		self._save_append(path,4)

		with h5py.File(path,'r') as h5file:
			group = h5file['DATASET']
			self.assertEqual(group['VARIABLES/time/value'].shape,(6,))
			self.assertEqual(group['FIELDS/velocity/value'].shape,(self.NPOINTS,6))
			self.assertEqual(int(group.attrs['appendCursor']),6)
			self.assertEqual(int(group['FIELDS/velocity/vars'][0]),6)

		self._assert_values(pyLOM.Dataset.load(path,mpio=False),0,6)

	def test_resizable_append_dispatch_is_opt_in(self):
		path = self._path('existing_append.h5')
		with patch.object(dataset_module.io,'h5_append_dset') as append_existing, \
			 patch.object(dataset_module.io,'h5_append_dset_resizable') as append_resizable:
			self._dataset(0).save(path,append=True,mpio=False)
		append_existing.assert_called_once()
		append_resizable.assert_not_called()

		with patch.object(dataset_module.io,'h5_append_dset') as append_existing, \
			 patch.object(dataset_module.io,'h5_append_dset_resizable') as append_resizable:
			self._dataset(0).save(path,append=True,append_resizable=True,mpio=False)
		append_existing.assert_not_called()
		append_resizable.assert_called_once()

	def test_different_block_size_is_rejected(self):
		path = self._path('block_size.h5')
		self._save_append(path,0)
		candidate = self._dataset(2,ninstants=1)
		layout_hash = h5io.h5_resizable_append_layout_hash(candidate.xyz,candidate.ordering)
		original_raise_error = h5io.raiseError
		def raise_value_error(message): raise ValueError(message)
		h5io.raiseError = raise_value_error
		try:
			with h5py.File(path,'r') as h5file:
				with self.assertRaisesRegex(ValueError,'block size differs'):
					h5io.h5_validate_resizable_append_group(
						h5file['DATASET'],candidate.xyz,candidate.vars,candidate.fields,
						candidate.ordering,candidate.point,candidate.partition_table,
						False,1,layout_hash,
					)
		finally:
			h5io.raiseError = original_raise_error

	def test_different_partition_layout_is_rejected(self):
		path = self._path('layout.h5')
		self._save_append(path,0)
		candidate = self._dataset(2)
		candidate._xyz   = candidate.xyz[::-1].copy()
		candidate._order = candidate.ordering[::-1].copy()
		for field in candidate.fields.values(): field['value'] = field['value'][::-1].copy()
		layout_hash = h5io.h5_resizable_append_layout_hash(candidate.xyz,candidate.ordering)
		original_raise_error = h5io.raiseError
		def raise_value_error(message): raise ValueError(message)
		h5io.raiseError = raise_value_error
		try:
			with h5py.File(path,'r') as h5file:
				with self.assertRaisesRegex(ValueError,'spatial layout differs'):
					h5io.h5_validate_resizable_append_group(
						h5file['DATASET'],candidate.xyz,candidate.vars,candidate.fields,
						candidate.ordering,candidate.point,candidate.partition_table,
						False,2,layout_hash,
					)
		finally:
			h5io.raiseError = original_raise_error

	def test_cursor_persists_when_files_are_appended_interleaved(self):
		path_a = self._path('interleaved_a.h5')
		path_b = self._path('interleaved_b.h5')
		self._save_append(path_a,0,total_size=4)
		self._save_append(path_b,100,total_size=4)
		self._save_append(path_a,2)
		self._save_append(path_b,102)
		with h5py.File(path_a,'r') as h5file:
			self.assertEqual(int(h5file['DATASET'].attrs['appendCursor']),4)
		with h5py.File(path_b,'r') as h5file:
			self.assertEqual(int(h5file['DATASET'].attrs['appendCursor']),4)

		self._assert_values(pyLOM.Dataset.load(path_a,mpio=False),0,4)
		self._assert_values(pyLOM.Dataset.load(path_b,mpio=False),100,4)

	def test_serial_nopartition_uses_global_ordering(self):
		path = self._path('nopartition.h5')
		permutation = np.array([4,2,0,3,1],dtype=np.int32)
		time = np.arange(2,dtype=np.float64)
		point = permutation.astype(np.float64)[:,None]
		dataset = pyLOM.Dataset(
			xyz=self.xyz[permutation],
			ptable=self.ptable,
			order=permutation,
			point=True,
			vars={
				'time':{'idim':0,'value':time},
				'step':{'idim':0,'value':time+1000.},
			},
			velocity={'ndim':1,'value':100.*point + time[None,:]},
			pressure={'ndim':1,'value':-10.*point - time[None,:]},
		)
		dataset.save(path,append=True,append_resizable=True,mpio=False,nopartition=True)

		with h5py.File(path,'r') as h5file:
			self.assertTrue(bool(h5file['DATASET'].attrs['appendNoPartition']))
		self._assert_values(pyLOM.Dataset.load(path,mpio=False),0,2)

	def test_ordinary_non_append_save_load_is_unchanged(self):
		path = self._path('ordinary.h5')
		dataset = self._dataset(0,6)
		dataset.save(path,mode='w',mpio=False)

		with h5py.File(path,'r') as h5file:
			group = h5file['DATASET']
			self.assertIn('VARIABLES',group)
			self.assertIn('FIELDS',group)
			self.assertEqual(group['VARIABLES/time/value'].shape,(6,))
			self.assertEqual(group['FIELDS/velocity/value'].shape,(self.NPOINTS,6))

		self._assert_values(pyLOM.Dataset.load(path,mpio=False),0,6)


if __name__ == '__main__':
	unittest.main()
