#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# POD general utilities.
#
# Last rev: 29/10/2021
from __future__ import print_function, division

import numpy as np

from ..         import inp_out as io
from ..utils.cr import cr_start, cr_stop


def extract_modes(U,ivar,npoints,modes=[],reshape=True):
	'''
	Extract modes for a certain variables
	'''
	cr_start('POD.extract_modes',0)
	nvars = U.shape[0]//npoints
	# Define modes to extract
	if len(modes) == 0: modes = np.arange(1,U.shape[1]+1,dtype=np.int32)
	# Allocate output array
	out =np.zeros((npoints,len(modes)),np.double)
	for m in modes:
		out[:,m-1] = U[ivar-1:nvars*npoints:nvars,m-1]
	# Return reshaped output
	cr_stop('POD.extract_modes',0)
	return out.reshape((len(modes)*npoints,),order='C') if reshape else out


def save(fname,U,S,V,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store POD variables in serial or parallel
	according to the partition used to compute the POD.
	'''
	cr_start('POD.save',0)
	io.h5_save_POD(fname,U,S,V,ptable,nvars=nvars,pointData=pointData,mode=mode)
	cr_stop('POD.save',0)


def load(fname,vars=['U','S','V'],ptable=None):
	'''
	Load POD variables in serial or parallel
	according to the partition used to compute the POD.
	'''
	cr_start('POD.load',0)
	varList = io.h5_load_POD(fname,vars,ptable)
	cr_stop('POD.load',0)
	return varList