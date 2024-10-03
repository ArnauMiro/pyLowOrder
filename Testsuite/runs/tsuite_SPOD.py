#!/usr/bin/env python
#
# PYLOM Testsuite
# Run SPOD on the cylinder dataset
#
# Last revision: 17/03/2023
from __future__ import print_function, division

import sys, os, json
import pyLOM


## Parameters
DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
PARAMS    = json.loads(str(sys.argv[4]).replace("'",'"'))


## Data loading
m     = pyLOM.Mesh.load(DATAFILE)
d     = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
X     = d.X(*VARIABLES)
t     = d.get_variable('time')
npwin = PARAMS['npwin'] #Number of snapshots in each window
nolap = PARAMS['nolap'] #Number of overlapping snapshots between windows


## Run SPOD
L, P, f = pyLOM.SPOD.run(X,t,nDFT=npwin,nolap=nolap,remove_mean=True)
if pyLOM.utils.is_rank_or_serial(root=0): 
    fig,_ = pyLOM.SPOD.plotSpectra(f, L)
    os.makedirs(OUTDIR,exist_ok=True)
    fig.savefig(f'{OUTDIR}/spectra.png',dpi=300)
pyLOM.SPOD.save(f'{OUTDIR}/results.h5',L,P,f,d.partition_table,nvars=len(VARIABLES),pointData=d.point)


## Testsuite output
M = pyLOM.SPOD.extract_modes(L,P,1,len(d),modes=[1,2,3])
pyLOM.pprint(0,'TSUITE X     =',X.min(),X.max(),X.mean())
pyLOM.pprint(0,'TSUITE L     =',L.min(),L.max(),L.mean())
pyLOM.pprint(0,'TSUITE p     =',P.min(),P.max(),P.mean())
pyLOM.pprint(0,'TSUITE f     =',f.min(),f.max(),f.mean())
pyLOM.pprint(0,'TSUITE modes =',M.min(),M.max(),M.mean())


## Dump to ParaView
# Spatial modes
d.add_field('spatial_modes',3,pyLOM.SPOD.extract_modes(L,P,1,len(d),modes=[1,2,3]))
pyLOM.io.pv_writer(m,d,'modes',basedir=f'{OUTDIR}/modes',instants=[0],times=[0.],vars=['spatial_modes'],fmt='vtkh5')


## Show and print timings
pyLOM.cr_info()
pyLOM.pprint(0,'End of output')