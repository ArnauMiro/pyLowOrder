#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# SPOD general utilities.
#
# Last rev: 17/03/2023
from __future__ import print_function, division

import numpy as np

from ..utils.gpu import cp
from ..          import inp_out as io, PartitionTable
from ..utils     import cr_nvtx as cr, gpu_to_cpu

@cr('SPOD.extract_modes')
def extract_modes(L:np.ndarray,P:np.ndarray,ivar:int,npoints:int,iblock:int=1,modes:list=[],reshape:bool=True):
	r'''
	When performing POD of several variables simultaneously, this function separates the spatial modes from each of the variables.

	Args:
		L (np.ndarray): modal energy spectra
		P (np.ndarray): SPOD spatial modes
		ivar (int): ID of the variable (i. e.) position in which it was concatenated to the rest of data (min=1, max=number of concatenated variables)
		npoints (int): number of points in the domain per variable
		modes (list, optional): list containing the id of the modes to separate (default ``[]``).
		iblock (int, optional): block ID which we want to extract the modes from.
		reshape (bool, optional): if true the output will be given as (len(modes)*npoints,) if not it the result will be (npoints, len(modes)) (default `` True ``)

	Returns:
		np.ndarray: modes of the variable ivar
	'''
	p = cp if type(L) is cp.ndarray else np
	nblocks = L.shape[1]
	nvars   = P.shape[0]//(npoints*nblocks)
	# Define modes to extract
	if len(modes) == 0: modes = p.arange(1,P.shape[1]+1,dtype=p.int32)
	# Allocate output array
	out = p.zeros((npoints,len(modes)),L.dtype)
	for m in modes:
		# Extract the block
		Pb         = P[(iblock-1)*nvars*npoints:iblock*nvars*npoints]
		out[:,m-1] = Pb[ivar-1:nvars*npoints:nvars,m-1]
	# Return reshaped output
	return out.reshape((len(modes)*npoints,),order='C') if reshape else out


@cr('SPOD.save')
def save(fname:str,L:np.ndarray,P:np.ndarray,f:np.ndarray,ptable:PartitionTable,nvars:int=1,pointData:bool=True,mode:str='w'):
	r'''
	Store SPOD results in serial or parallel according to the partition used to compute the SPOD. It will be saved on a h5 file.

	Args:
		fname (str): path to the .h5 file in which the SPOD will be saved
		L (np.ndarray): modal energy
		P (np.ndarray): SPOD spatial modes
		f (np.ndarray): modes frequencies
		ptable (PartitionTable): partition table used to compute the SPOD
		nvars (int, optional): number of concatenated variables when computing the SPOD (default ``1``)
		pointData (bool, optional): bool to specify if the SPOD was performed either on point data or cell data (default ``True``)
		mode (str, optional): mode in which the HDF5 file is opened, 'w' stands for write mode and 'a' stands for append mode. Write mode will overwrite the file and append mode will add the informaiton at the end of the current file, choose with great care what to do in your case (default ``w``).

	'''
	io.h5_save_SPOD(fname,gpu_to_cpu(L),gpu_to_cpu(P),gpu_to_cpu(f),ptable,nvars=nvars,pointData=pointData,mode=mode)


@cr('SPOD.load')
def load(fname:str,vars:list=['L','P','f'],nmod:int=-1,ptable:PartitionTable=None):
	r'''
	Load SPOD results from a .h5 file in serial or parallel according to the partition used to compute the SPOD.

	Args:
		fname (str): path to the .h5 file in which the SPOD was saved
		vars (list): list of variables to load. The following notation, consistent with the save function, is used,
			'L': modes energy
			'P': spatial modes
			'f': modes frequency
		the default option is to load them all, but it is not recommended to load the spatial modes if they are not going to be used during the rest of the script.
		nmod (int, optional): number of modes to load. By default it will load all the saved modes (default, ``-1``)
		ptable (PartitionTable, optional): partition table to use when loading the data (default ``None``).
	'''
	return io.h5_load_SPOD(fname,vars,nmod,ptable)