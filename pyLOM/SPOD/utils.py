#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# SPOD general utilities.
#
# Last rev: 17/03/2023
from __future__ import print_function, division

import numpy as np

from ..         import inp_out as io
from ..utils.cr import cr_start, cr_stop


def extract_modes(L,P,ivar,npoints,iblock=1,modes=[],reshape=True):
	'''
	Extract modes for a certain variables
	'''
	cr_start('SPOD.extract_modes',0)
	nblocks = L.shape[1]
	nvars   = P.shape[0]//(npoints*nblocks)
	# Define modes to extract
	if len(modes) == 0: modes = np.arange(1,P.shape[1]+1,dtype=np.int32)
	# Allocate output array
	out =np.zeros((npoints,len(modes)),np.double)
	for m in modes:
		# Extract the block
		Pb         = P[(iblock-1)*nvars*npoints:iblock*nvars*npoints]
		out[:,m-1] = Pb[ivar-1:nvars*npoints:nvars,m-1]
	# Return reshaped output
	cr_stop('SPOD.extract_modes',0)
	return out.reshape((len(modes)*npoints,),order='C') if reshape else out