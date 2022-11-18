#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# DMD plotting utilities.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from ..vmmath      import fft
from ..utils.plots import plotResidual, plotFieldStruct2D, plotSnapshot, animateFlow
from ..utils.mesh  import mesh_compute_cellcenter


def plotMode(P, freqs, xyz,mesh,info,dim=0, f2plot = np.array([1],np.int32), modes=np.array([1],np.int32),scale_freq=1.,fig=[],ax=[],cmap=None):
	'''
	Given P and the frequencies, plot the requested modes
	'''
	cf = []
	for iif, f in enumerate(f2plot):
		if len(fig) < iif + 1:
			fig.append( plt.figure(figsize=(8,6),dpi=100) )
		if len(ax) < iif + 1:
			ax.append( fig[iif].subplots(modes.shape[0],1,gridspec_kw = {'hspace':0.5}) )
			fig[iif].suptitle('St = %.3f' % (np.abs(freqs[f-1])))
		for imode, mode in enumerate(modes):
			if mesh['type'] == 'struct2D':
				c = None
				if info['point']:
					M  = mesh['nx']*mesh['ny']
					c1 = plotFieldStruct2D(ax[iif][imode],mesh['nx'],mesh['ny'],info['ndim'],xyz,P[(mode-1)*M:mode*M,f-1],dim-1,cmap)
				else:
					xyzc = mesh_compute_cellcenter(xyz,mesh)
					M  = (mesh['nx']-1)*(mesh['ny']-1)
					c1 = plotFieldStruct2D(ax[iif][imode],mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,P[(mode-1)*M:mode*M,f-1],dim-1,cmap)
				cf.append(c)
			plt.colorbar(mappable = c1, ax=ax[iif][imode])
			ax[iif][imode].set_title('Mode %i' % (mode))
			ax[iif][imode].set_xlabel('x/D')
			ax[iif][imode].set_ylabel('y/D')
			ax[iif][imode].set_aspect('equal')
	return fig, ax, cf

def plotSpectra(f, L):
	for ii in range(L.shape[1]):
		plt.loglog(np.sort(f), L[np.argsort(f),ii], 'o-')
	plt.xlabel('St')
	plt.ylabel(r'$\lambda_i$')