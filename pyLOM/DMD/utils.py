#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# DMD general utilities.
#
# Last rev: 27/01/2023
from __future__ import print_function, division

import numpy as np

from ..         import inp_out as io
from ..utils.cr import cr_start, cr_stop


def extract_modes(Phi,ivar,npoints,real=True,modes=[],reshape=True):
	'''
	Extract modes for a certain variables
	'''
	cr_start('DMD.extract_modes',0)
	nvars = Phi.shape[0]//npoints
	# Define modes to extract
	if len(modes) == 0: modes = np.arange(1,Phi.shape[1]+1,dtype=np.int32)
	# Allocate output array
	out =np.zeros((npoints,len(modes)),np.double)
	for m in modes:
		out[:,m-1] = Phi[ivar-1:nvars*npoints:nvars,m-1].real if real else Phi[ivar-1:nvars*npoints:nvars,m-1].imag
	# Return reshaped output
	cr_stop('DMD.extract_modes',0)
	return out.reshape((len(modes)*npoints,),order='C') if reshape else out


def save(fname,muReal,muImag,Phi,bJov,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store DMD variables in serial or parallel
	according to the partition used to compute the DMD.
	'''
	cr_start('DMD.save',0)
	io.h5_save_DMD(fname,muReal,muImag,Phi,bJov,ptable,nvars=nvars,pointData=pointData,mode=mode)
	cr_stop('DMD.save',0)


def load(fname,vars=['Phi','mu','bJov','delta','omega'],ptable=None):
	'''
	Load DMD variables in serial or parallel
	according to the partition used to compute the DMD.
	'''
	cr_start('DMD.load',0)
	varList = io.h5_load_DMD(fname,vars,ptable)
	cr_stop('DMD.load',0)
	return varList