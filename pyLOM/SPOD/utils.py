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
from ..          import inp_out as io
from ..utils.cr  import cr_nvtx as cr


@cr('SPOD.extract_modes')
def extract_modes(L,P,ivar,npoints,iblock=1,modes=[],reshape=True):
	'''
	Extract modes for a certain variables
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
def save(fname,L,P,f,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store SPOD variables in serial or parallel
	according to the partition used to compute the SPOD.
	'''
	if type(L) is cp.ndarray:
		io.h5_save_SPOD(fname,L.get(),P.get(),f.get(),ptable,nvars=nvars,pointData=pointData,mode=mode)
	else:
		io.h5_save_SPOD(fname,L,P,f,ptable,nvars=nvars,pointData=pointData,mode=mode)


@cr('SPOD.load')
def load(fname,vars=['L','P','f'],nmod=-1,ptable=None):
	'''
	Load SPOD variables in serial or parallel
	according to the partition used to compute the SPOD.
	'''
	return io.h5_load_SPOD(fname,vars,nmod,ptable)