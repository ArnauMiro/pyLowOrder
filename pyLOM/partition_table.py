#!/usr/bin/env python
#
# pyLOM, dataset.
#
# Dataset class, reader and reduction routines.
#
# Last rev: 30/07/2021
from __future__ import print_function, division

import numpy as np

from .utils.parall import MPI_SIZE, worksplit, mpi_gather
from .utils.cr     import cr
from .utils.mem    import mem


class PartitionTable(object):
	'''
	The partition table class contains information on the 
	partition used for the given dataset or  it can generate
	a new partition
	'''
	@mem('PartTable')
	def __init__(self,nparts,ids,elements,points,has_master=False):
		'''
		Class constructor
		'''
		self._nparts      = nparts
		self._ids         = ids
		self._elements    = elements
		self._master      = has_master if MPI_SIZE > 1 else False
		self._points      = points

	def __str__(self):
		out  = 'Partition Table:\nnumber of partitions: %d\n' % self.n_partitions
		out += '\tIds  |  Elements  |  Points  \n'
		for ipart in range(self.n_partitions):
			out += '\t %03d |    %04d    |    %04d \n' %(self.Ids[ipart],self.Elements[ipart],self.Points[ipart])
		return out

	@cr('PartTable.pbounds')
	def partition_bounds(self,rank,ndim=1,points=True):
		'''
		Compute the partition bounds for a given rank
		'''
		if self._master and rank == 0 and not MPI_SIZE == 1: 
			return 0, 1
		offst    = 1 if not self._master else 0
		mask_idx = self.Ids < rank + offst
		this_idx = self.Ids == rank + offst
		table    = self.Points if points else self.Elements
		istart   = np.sum(table[mask_idx])*ndim
		iend     = istart + table[this_idx][0]*ndim
		return istart, iend

	@cr('PartTable.ppoints')
	def partition_points(self,rank,npoints,conec,ndim=1):
		'''
		Compute the points to be read for this partition
		'''
		# Find which nodes this partition has
		thenods = np.unique(conec.flatten())
		mynods  = np.array([],np.int32)
		# Deal with multiple dimensions
		for idim in range(ndim):
			mynods = np.hstack((mynods,thenods+idim*npoints))
		return mynods		

	@cr('PartTable.reorder')
	def reorder_points(self,xyz,conectivity):
		'''
		Reorder the points array so that in matches with
		the partition table, in serial algorithm.
		'''
		xyz_new = np.zeros_like(xyz)
		# Loop all the partitions
		for ipart in range(self.n_partitions):
			mynods = self.partition_points(ipart,conectivity)
			# Rearrange the node vector
			nstart  = np.cumsum(self.Points[:ipart])
			nend    = self.Points[ipart] + nstart
			xyz_new[nstart:nend] = xyz[mynods]
		# Return
		return xyz_new

	def update_points(self,npoints_new):
		'''
		Update the number of points on the table
		'''
		p = mpi_gather(npoints_new,all=True)
		self._points = p if isinstance(p,np.ndarray) else np.array([p],np.int32)

	def check_split(self):
		'''
		See if a table has the same number of subdomains
		than the number of mpi ranks
		'''
		# Deal with master and serial
		offst = 1 if self._master and not MPI_SIZE == 1 else 0
		return self._nparts + offst == MPI_SIZE

	@classmethod
	@cr('PartTable.new')
	def new(cls,nparts,nelems,npoints,has_master=False):
		'''
		Create a new partition table, in serial algorithm.
		'''
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
		return cls(nparts,ids,elements,points,has_master=has_master)

	@classmethod
	@cr('PartTable.from_pyAlya')
	def from_pyAlya(cls,ptable,has_master=True):
		'''
		Create a partition table from a partition table coming
		from Alya
		'''
		nparts   = ptable.n_partitions
		ids      = np.arange(1,nparts+1,dtype=np.int32)
		points   = ptable.Points
		elements = ptable.Elements
		return cls(nparts,ids,elements,points,has_master=has_master)		

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
	@property
	def has_master(self):
		return self._master