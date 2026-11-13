#!/usr/bin/env python
#
# Example of SPOD convergence study: the leading modes of a block computed
# with the full dataset are compared against the same modes computed with
# the first and the last 75% of the snapshots.
#
# Last revision: 13/11/2026
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np

import pyLOM

pyLOM.gpu_device(gpu_per_node=4)


## Parameters
DATAFILE = './DATA/CYLINDER.h5'
VARIABLE = 'VELOC'
FRAC     = 0.75 #Fraction of snapshots kept in the reduced datasets


## Data loading
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table).to_gpu([VARIABLE])
X = d[VARIABLE]
t = d.get_variable('time')
N = X.shape[1]


## Window parameters
#nDFT and nolap must be the same in the three runs so that the frequencies
#coincide. They are computed once from the full dataset and reused: with
#nDFT = 0 each run would compute its own value, which depends on the number
#of snapshots and would differ between the full and the reduced datasets.
nDFT  = N//2     #Number of snapshots in each window
nolap = nDFT//2  #Number of overlapping snapshots between windows


## Run SPOD
#Reference run on the full dataset and reduced runs on the first and the
#last FRAC fraction of the snapshots
i75      = int(FRAC*N)
Lr, Pr, fr = pyLOM.SPOD.run(X,          t,          nDFT=nDFT, nolap=nolap, remove_mean=True)
L1, P1, f1 = pyLOM.SPOD.run(X[:, :i75], t[:i75],    nDFT=nDFT, nolap=nolap, remove_mean=True)
L2, P2, f2 = pyLOM.SPOD.run(X[:, N-i75:], t[N-i75:], nDFT=nDFT, nolap=nolap, remove_mean=True)


## Extract the leading block of modes
#The columns of P are sorted by decreasing energy independently in each run,
#so column j may correspond to a different frequency in each run. Check
#whether the frequencies appear in the same order; if not, realign the mode
#ids of the reduced runs with the reference frequencies.
def col_of(freqs, f):
	'''1-based id of the mode column at frequency f'''
	return int(np.where(np.isclose(freqs, f))[0][0]) + 1

modes  = list(range(1, fr.shape[0]+1)) #Ids of all the frequencies, reference order
modes1 = modes if np.allclose(f1, fr) else [col_of(f1, v) for v in fr]
modes2 = modes if np.allclose(f2, fr) else [col_of(f2, v) for v in fr]

Pr_ = pyLOM.SPOD.extract_modes(Lr, Pr, 1, len(d), iblock=1, modes=modes,  reshape=False)
P1_ = pyLOM.SPOD.extract_modes(L1, P1, 1, len(d), iblock=1, modes=modes1, reshape=False)
P2_ = pyLOM.SPOD.extract_modes(L2, P2, 1, len(d), iblock=1, modes=modes2, reshape=False)


## Plot the convergence of the modes
#The modes are stored as real arrays, so the convergence is measured with the
#absolute value of the cosine of the angle between the reference and the
#reduced modes. Note that the windows of the last reduced run are shifted in
#time by tau, so its curve is expected to show mu ~ |cos(2*pi*f*tau)| at each
#frequency f.
if pyLOM.utils.is_rank_or_serial(root=0):
	pyLOM.SPOD.plotConvergence(Pr_, P1_, P2_, f=fr)


## Show and print timings
pyLOM.cr_info()
pyLOM.show_plots()
