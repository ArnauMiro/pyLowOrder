#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# POD plotting utilities.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from ..vmmath      import fft
from ..utils.plots import plotResidual, plotFieldStruct2D, plotSnapshot, animateFlow
from ..utils.mesh  import mesh_compute_cellcenter

def plotMode(U,xyz,V,t,mesh,info,dim=0,modes=np.array([1],np.int32),scale_freq=1.,fig=[],ax=[],cmap=None):
	'''
	Given U, VT and a mode, plot their
	representation in a figure.
	'''
	cf = []
	for imode, mode in enumerate(modes):
		if len(fig) < imode + 1:
			fig.append( plt.figure(figsize=(8,6),dpi=100) )
		if len(ax) < imode + 1:
			ax.append( fig[imode].subplots(3,1,gridspec_kw = {'hspace':0.5}) )
		fig[imode].suptitle('Mode %d'%(mode-1))
	   	# Plot the representation of mode U
		if mesh['type'] == 'struct2D':
			c = None
			if info['point']:
				c = plotFieldStruct2D(ax[imode][0],mesh['nx'],mesh['ny'],info['ndim'],xyz,U[:,mode-1],dim-1,cmap)
			else:
				xyzc = mesh_compute_cellcenter(xyz,mesh)
				c = plotFieldStruct2D(ax[imode][0],mesh['nx']-1,mesh['ny']-1,info['ndim'],xyzc,U[:,mode-1],dim-1,cmap)
			cf.append(c)
		plt.colorbar(mappable = c, ax=ax[imode][0])
		ax[imode][0].set_title('Spatial mode')
		ax[imode][0].set_aspect('equal')
		ax[imode][0].set_xlim([-1,8])
		ax[imode][0].set_xlabel('x/D')
		ax[imode][0].set_ylabel('y/D')
	   	# Plot the temporal evolution of the mode
		ax[imode][1].plot(t,V[mode-1,:],'b')
		ax[imode][1].set_title('Temporal mode')
		# Plot frequency representation of the mode
		if V.shape[1] % 2 == 0:
			freq,psd = fft(t,V[mode-1,:],equispaced=False)
		else:
			freq,psd = fft(t[:-1],V[mode-1,:-1], equispaced=False)
		freq *= scale_freq
		#L = int(np.floor(freq.shape[0]/2))
		ax[imode][2].plot(freq,psd, 'b')
		ax[imode][2].set_title('Power Spectrum')
		ax[imode][2].set_xlabel('St')
		ax[imode][2].set_xlim([0,1])
	return fig, ax, cf
