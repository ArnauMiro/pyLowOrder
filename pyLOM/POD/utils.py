#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# POD general utilities.
#
# Last rev: 29/10/2021
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp
from ..          import inp_out as io, PartitionTable
from ..utils     import cr_nvtx as cr, gpu_to_cpu

@cr('POD.extract_modes')
def extract_modes(U:np.ndarray,ivar:int,npoints:int,modes:list=[],reshape:bool=True):
	r'''
	When performing POD of several variables simultaneously, this function separates the spatial modes from each of the variables.

	Args:
		U (np.ndarray): POD spatial modes
		ivar (int): ID of the variable (i. e.) position in which it was concatenated to the rest of data (min=1, max=number of concatenated variables)
		npoints (int): number of points in the domain per variable
		modes (list, optional): list containing the id of the modes to separate (default ``[]``).
		reshape (bool, optional): if true the output will be given as (len(modes)*npoints,) if not it the result will be (npoints, len(modes)) (default `` True ``)

	Returns:
		np.ndarray: modes of the variable ivar
	'''
	p = cp if type(U) is cp.ndarray else np
	nvars = U.shape[0]//npoints
	# Define modes to extract
	if len(modes) == 0: modes = p.arange(1,U.shape[1]+1,dtype=p.int32)
	# Allocate output array
	out =p.zeros((npoints,len(modes)),U.dtype)
	for i,m in enumerate(modes):
		out[:,i] = U[ivar-1:nvars*npoints:nvars,m-1]
	# Return reshaped output
	return out.reshape((len(modes)*npoints,),order='C') if reshape else out


@cr('POD.save')
def save(fname:str,U:np.ndarray,S:np.ndarray,V:np.ndarray,ptable:PartitionTable,nvars:int=1,pointData:bool=True,mode:str='w'):
	r'''
	Store POD results in serial or parallel according to the partition used to compute the POD. It will be saved on a h5 file.

	Args:
		fname (str): path to the .h5 file in which the POD will be saved
		U (np.ndarray): spatial modes to save. To avoid saving the spatial modes, just give None as input
		S (np.ndarray): singular values to save. To avoid saving the singular values, just give None as input
		V (np.ndarray): temporal coefficients to save. To avoid saving the temporal coefficients, just give None as input
		ptable (PartitionTable): partition table used to compute the POD
		nvars (int, optional): number of concatenated variables when computing the POD (default ``1``)
		pointData (bool, optional): bool to specify if the POD was performed either on point data or cell data (default ``True``)
		mode (str, optional): mode in which the HDF5 file is opened, 'w' stands for write mode and 'a' stands for append mode. Write mode will overwrite the file and append mode will add the informaiton at the end of the current file, choose with great care what to do in your case (default ``w``).

	'''
	io.h5_save_POD(fname,gpu_to_cpu(U),gpu_to_cpu(S),gpu_to_cpu(V),ptable,nvars=nvars,pointData=pointData,mode=mode)


@cr('POD.load')
def load(fname:str,vars:list=['U','S','V'],nmod:int=-1,ptable:PartitionTable=None):
	r'''
	Load POD results from a .h5 file in serial or parallel according to the partition used to compute the POD.

	Args:
		fname (str): path to the .h5 file in which the POD was saved
		vars (list): list of variables to load. The following notation, consistent with the save function, is used,
			'U': spatial modes
			'S': singular values
			'V': temporal coefficients
		the default option is to load them all, but it is not recommended to load the spatial modes if they are not going to be used during the rest of the script.
		nmod (int, optional): number of modes to load. By default it will load all the saved modes (default, ``-1``)
		ptable (PartitionTable, optional): partition table to use when loading the data (default ``None``).
	'''
	return io.h5_load_POD(fname,vars,nmod,ptable)
