#!/usr/bin/env python
#
# pyLOM, dataset.
#
# Dataset class, reader and reduction routines.
#
# Last rev: 30/07/2021
from __future__ import print_function, division

import numpy as np

from .utils.parall import worksplit, mpi_gather
from .utils.cr     import cr_start, cr_stop


class PartitionTable(object):
	'''
	The partition table class contains information on the 
	partition used for the given dataset or  it can generate
	a new partition
	'''
	def __init__(self,nparts,ids,elements,points):
		'''
		Class constructor
		'''
		self._nparts      = nparts
		self._ids         = ids
		self._elements    = elements
		self._points      = points

	def __str__(self):
		out  = 'Partition Table:\nnumber of partitions: %d\n' % self.n_partitions
		out += '\tIds  |  Elements  |  Points  \n'
		for ipart in range(self.n_partitions):
			out += '\t %03d |    %04d    |    %04d \n' %(self.Ids[ipart],self.Elements[ipart],self.Points[ipart])
		return out

	def partition_bounds(self,rank,ndim=1,points=True):
		'''
		Compute the partition bounds for a given rank
		'''
		cr_start('ptable part bound',0)
		mask_idx = self.Ids < rank + 1
		this_idx = self.Ids == rank + 1
		table    = self.Points if points else self.Elements
		istart   = np.sum(table[mask_idx])*ndim
		iend     = istart + table[this_idx][0]*ndim
		cr_stop('ptable part bound',0)
		return istart, iend

	def partition_points(self,rank,npoints,conec,ndim=1):
		'''
		Compute the points to be read for this partition
		'''
		cr_start('ptable part point',0)
		# Find which nodes this partition has
		thenods = np.unique(conec.flatten())
		mynods  = np.array([],np.int32)
		# Deal with multiple dimensions
		for idim in range(ndim):
			mynods = np.hstack((mynods,thenods+idim*npoints))
		cr_stop('ptable part point',0)
		return mynods		

	def reorder_points(self,xyz,conectivity):
		'''
		Reorder the points array so that in matches with
		the partition table, in serial algorithm.
		'''
		cr_start('ptable reorder',0)
		xyz_new = np.zeros_like(xyz)
		# Loop all the partitions
		for ipart in range(self.n_partitions):
			mynods = self.partition_points(ipart,conectivity)
			# Rearrange the node vector
			nstart  = np.cumsum(self.Points[:ipart])
			nend    = self.Points[ipart] + nstart
			xyz_new[nstart:nend] = xyz[mynods]
		# Return
		cr_stop('ptable reorder',0)
		return xyz_new

	def update_points(self,npoints_new):
		'''
		Update the number of points on the table
		'''
		p = mpi_gather(npoints_new,all=True)
		self._points = p if isinstance(p,np.ndarray) else np.array([p],np.int32)

	@classmethod
	def new(cls,nparts,nelems,npoints):
		'''
		Create a new partition table, in serial algorithm.
		'''
		cr_start('ptable new',0)
		ids      = np.zeros((nparts,),np.int32)
		points   = np.zeros((nparts,),np.int32)
		elements = np.zeros((nparts,),np.int32)
		# For all the partitions do
		for ipart in range(nparts):
			ids[ipart] = ipart + 1
			# Split the number of elements
			istart, iend = worksplit(0,nelems,ipart,nWorkers=nparts)
			# How many elements do I have
			elements[ipart] = iend - istart
			# How many nodes do I have
			istart, iend  = worksplit(0,npoints,ipart,nWorkers=nparts)
			points[ipart] = iend - istart
		cr_stop('ptable new',0)
		return cls(nparts,ids,elements,points)

	@property
	def n_partitions(self):
		return self._nparts
	@property
	def Ids(self):
		return self._ids
	@property
	def Elements(self):
		return self._elements
	@property
	def Points(self):
		return self._points