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
from ..utils.cr import cr


@cr('DMD.extract_modes')
def extract_modes(Phi,ivar,npoints,real=True,modes=[],reshape=True):
	'''
	Extract modes for a certain variables
	'''
	nvars = Phi.shape[0]//npoints
	# Define modes to extract
	if len(modes) == 0: modes = np.arange(1,Phi.shape[1]+1,dtype=np.int32)
	# Allocate output array
	out =np.zeros((npoints,len(modes)),np.double)
	for i,m in enumerate(modes):
		out[:,i] = Phi[ivar-1:nvars*npoints:nvars,m-1].real if real else Phi[ivar-1:nvars*npoints:nvars,m-1].imag
	# Return reshaped output
	return out.reshape((len(modes)*npoints,),order='C') if reshape else out


@cr('DMD.save')
def save(fname,muReal,muImag,Phi,bJov,ptable,nvars=1,pointData=True,mode='w'):
	'''
	Store DMD variables in serial or parallel
	according to the partition used to compute the DMD.
	'''
	io.h5_save_DMD(fname,muReal,muImag,Phi,bJov,ptable,nvars=nvars,pointData=pointData,mode=mode)


@cr('DMD.load')
def load(fname,vars=['Phi','mu','bJov','delta','omega'],nmod=-1,ptable=None):
	'''
	Load DMD variables in serial or parallel
	according to the partition used to compute the DMD.
	'''
	return io.h5_load_DMD(fname,vars,nmod,ptable)
