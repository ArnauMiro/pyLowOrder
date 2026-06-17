#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# DMD general utilities.
#
# Last rev: 27/01/2023
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp
from ..          import inp_out as io
from ..utils     import cr_nvtx as cr, gpu_to_cpu


@cr('DMD.extract_modes')
def extract_modes(Phi,ivar,npoints,real=True,modes=[],reshape=True):
	'''
	Extract modes for a certain variables.

	Args:
		Phi (np.ndarray or cp.ndarray) : DMD modes.
		ivar (int): Variable to extract.
		npoints (int): Number of points of the mesh.
		real (bool, optional) : If ``True`` (as default) returns the real part
			of the modes, otherwise the imaginary part.
		modes (list, optional) : List of modes to extract. The default is the
			empty list.
		reshape (bool, optional) : If ``True`` (as default) returns the output
			as a vector, otherwise the modes form the columns of the return
			matrix.

	Returns:
		np.ndarray:
			Requested modes.
	'''
	p = cp if type(Phi) is cp.ndarray else np
	nvars = Phi.shape[0]//npoints
	# Define modes to extract
	if len(modes) == 0: modes = p.arange(1,Phi.shape[1]+1,dtype=p.int32)
	# Allocate output array
	out =p.zeros((npoints,len(modes)),p.double if Phi.dtype == p.complex128 else p.float32)
	for i,m in enumerate(modes):
		out[:,i] = Phi[ivar-1:nvars*npoints:nvars,m-1].real if real else Phi[ivar-1:nvars*npoints:nvars,m-1].imag
	# Return reshaped output
	return out.reshape((len(modes)*npoints,),order='C') if reshape else out


@cr('DMD.save',color='blue')
def save(fname,muReal,muImag,Phi,bJov,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store DMD variables in serial or parallel
	according to the partition used to compute the DMD.

	Args:
		fname (str) : File name.
		muReal (np.ndarray or cp.ndarray) : Real part of the eigenvalues.
		muImag (np.ndarray or cp.ndarray) : Imaginary part of the eigenvalues.
		Phi (np.ndarray or cp.ndarray) : DMD modes.
		bJov (np.ndarray or cp.ndarray) : Amplitude of the DMD modes.
		ptable (pyLOM.PartitionTable) : Partition table.
		nvars (int, optional) : Number of variables of the field.
		pointData (bool, optional) : ``True`` if the data corresponds to the
			nodes of the mesh, ``False`` if it corresponds to the cells.
		mode (str, optional) : Writing mode. Set to ``'a'`` to append to another
			hdf file. Then no partition table will be saved.
	'''
	io.h5_save_DMD(fname,gpu_to_cpu(muReal),gpu_to_cpu(muImag),gpu_to_cpu(Phi),gpu_to_cpu(bJov),ptable,nvars=nvars,pointData=pointData,mode=mode)


@cr('DMD.load',color='blue')
def load(fname,vars=['Phi','mu','bJov','delta','omega'],nmod=-1,ptable=None):
	'''
	Load DMD variables in serial or parallel
	according to the partition used to compute the DMD.

	Args:
		fname (str) : File name.
		vars (list, optional) : List of strings to be loaded.
		nmod (int, optional) : Apparently unused argument.
		ptable (pyLOM.PartitionTable, optional) : If given, the parition table
			will not be loaded from the file.

	Returns:
		list[np.ndarray]:
			List with the arrays of the requested variables.
	'''
	return io.h5_load_DMD(fname,vars,nmod,ptable)
