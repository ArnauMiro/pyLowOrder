#!/usr/bin/env python
#
# PYLOM Testsuite POD
#
# Last revision: 20/09/2024
from __future__ import print_function, division

import sys, os, json, numpy as np
import pyLOM


## Parameters
DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
PARAMS    = json.loads(str(sys.argv[4]).replace("'",'"').lower())

## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X = d.X(*VARIABLES)
t = d.get_variable('time')


## Run POD
PSI,S,V = pyLOM.POD.run(X,**PARAMS['run']) # PSI are POD modes
if pyLOM.utils.is_rank_or_serial(root=0): 
	fig,_ = pyLOM.POD.plotResidual(S)
	os.makedirs(OUTDIR,exist_ok=True)
	fig.savefig(f'{OUTDIR}/residuals.png',dpi=300)
# Truncate according to a residual
PSI,S,V = pyLOM.POD.truncate(PSI,S,V,r=PARAMS['r_trunc'])
print("KKK",PSI.shape)
pyLOM.POD.save(f'{OUTDIR}/results.h5',PSI,S,V,d.partition_table,nvars=len(VARIABLES),pointData=d.point)
# Reconstruct the flow
X_POD = pyLOM.POD.reconstruct(PSI,S,V)
# Compute RMSE
rmse = pyLOM.math.RMSE(X_POD,X)


## Testsuite output
M = pyLOM.POD.extract_modes(PSI,1,len(d),modes=PARAMS['modes'])
pyLOM.pprint(0,'TSUITE RMSE  = %e'%rmse)
pyLOM.pprint(0,'TSUITE X     =',X.min(),X.max(),X.mean())
pyLOM.pprint(0,'TSUITE S     =',S.min(),S.max(),S.mean())
pyLOM.pprint(0,'TSUITE V     =',V.min(),V.max(),V.mean())
pyLOM.pprint(0,'TSUITE X_POD =',X_POD.min(),X_POD.max(),X_POD.mean())
pyLOM.pprint(0,'TSUITE modes =',M.min(),M.max(),M.mean())


## Dump to ParaView
# Spatial modes
d.add_field('spatial_modes',len(PARAMS['modes']),pyLOM.POD.extract_modes(PSI,1,len(d),modes=PARAMS['modes']))
pyLOM.io.pv_writer(m,d,'modes',basedir=f'{OUTDIR}/modes',instants=[0],times=[0.],vars=['spatial_modes'],fmt='vtkh5')

# Temporal evolution
d.add_field('RECON',len(VARIABLES),X_POD)
pyLOM.io.pv_writer(m,d,'flow',basedir=f'{OUTDIR}/flow',instants=np.arange(t.shape[0],dtype=np.int32),times=t,vars=VARIABLES+['RECON'],fmt='vtkh5')


## Plot POD mode
if pyLOM.utils.is_rank_or_serial(0):
	# 0 - module, 1,2 - components
	fig, _ = pyLOM.POD.plotMode(V[:,:-1],t[:-1],modes=PARAMS['modes'])
	for i,f in enumerate(fig): f.savefig(f'{OUTDIR}/modes_%d.png'%i,dpi=300)


## Show and print timings
pyLOM.cr_info()
pyLOM.pprint(0,'End of output')