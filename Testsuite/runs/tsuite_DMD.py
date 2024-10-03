#!/usr/bin/env python
#
# PYLOM Testsuite
# Run DMD on the cylinder dataset
#
# Last revision: 27/01/2023
from __future__ import print_function, division

import sys, os, numpy as np
import pyLOM


## Parameters
DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]


## Data loading
m  = pyLOM.Mesh.load(DATAFILE)
d  = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d.X(*VARIABLES)
t  = d.get_variable('time')
dt = t[1] - t[0]


## Run DMD
Y = X[:,:100].copy() # Grab the first 100 and run DMD
muReal,muImag,Phi,bJov = pyLOM.DMD.run(Y,1e-6,remove_mean=False)
# Compute frequency and damping ratio of the modes
delta, omega = pyLOM.DMD.frequency_damping(muReal,muImag,dt)
os.makedirs(OUTDIR,exist_ok=True)
pyLOM.DMD.save(f'{OUTDIR}/results.h5',muReal,muImag,Phi,bJov,d.partition_table,nvars=len(VARIABLES),pointData=d.point)
# Reconstruction according to Jovanovic 2014
# Last 50 snapshots are predicted
X_DMD = pyLOM.DMD.reconstruction_jovanovic(Phi,muReal,muImag,t,bJov)
rmse  = pyLOM.math.RMSE(X_DMD.copy(),X.copy())


## Testsuite output
M_real = pyLOM.DMD.extract_modes(Phi,1,len(d),real=True,modes=[1,4,6,2,5,3])
M_imag = pyLOM.DMD.extract_modes(Phi,1,len(d),real=False,modes=[1,4,6,2,5,3])
pyLOM.pprint(0,'TSUITE RMSE   = %e'%rmse)
pyLOM.pprint(0,'TSUITE muReal =',muReal.min(),muReal.max(),muReal.mean())
pyLOM.pprint(0,'TSUITE muImag =',muImag.min(),muImag.max(),muImag.mean())
pyLOM.pprint(0,'TSUITE Phi    =',Phi.min(),Phi.max(),Phi.mean())
pyLOM.pprint(0,'TSUITE bJov   =',bJov.min(),bJov.max(),bJov.mean())
pyLOM.pprint(0,'TSUITE delta  =',delta.min(),delta.max(),delta.mean())
pyLOM.pprint(0,'TSUITE omega  =',omega.min(),omega.max(),omega.mean())
pyLOM.pprint(0,'TSUITE Y      =',Y.min(),Y.max(),Y.mean())
pyLOM.pprint(0,'TSUITE X_DMD  =',X_DMD.min(),X_DMD.max(),X_DMD.mean())
pyLOM.pprint(0,'TSUITE M_real =',M_real.min(),M_real.max(),M_real.mean())
pyLOM.pprint(0,'TSUITE M_imag =',M_imag.min(),M_imag.max(),M_imag.mean())


## DMD plots
if pyLOM.utils.is_rank_or_serial(0):
	fig,_ = pyLOM.DMD.ritzSpectrum(muReal,muImag)               # Ritz Spectrum
	fig.savefig(f'{OUTDIR}/ritzSpectrum.png',dpi=300)
	fig,_ = pyLOM.DMD.amplitudeFrequency(omega,bJov,norm=False) # Amplitude vs frequency
	fig.savefig(f'{OUTDIR}/amplitudeFrequency.png',dpi=300)
	fig,_ = pyLOM.DMD.dampingFrequency(omega,delta)             # Damping ratio vs frequency
	fig.savefig(f'{OUTDIR}/dampingFrequency.png',dpi=300)


## Dump to ParaView
# Spatial modes
d.add_field('MODES_REAL',6,pyLOM.DMD.extract_modes(Phi,1,len(d),real=True,modes=[1,4,6,2,5,3]))
d.add_field('MODES_IMAG',6,pyLOM.DMD.extract_modes(Phi,1,len(d),real=False,modes=[1,4,6,2,5,3]))
pyLOM.io.pv_writer(m,d,'modes',basedir=f'{OUTDIR}/modes',instants=[0],times=[0.],vars=['MODES_REAL','MODES_IMAG'],fmt='vtkh5')

# Temporal evolution
d.add_field('RECON',len(VARIABLES),X_DMD)
pyLOM.io.pv_writer(m,d,'flow',basedir=f'{OUTDIR}/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=VARIABLES+['RECON'],fmt='vtkh5')


## Show and print timings
pyLOM.cr_info()
pyLOM.pprint(0,'End of output')