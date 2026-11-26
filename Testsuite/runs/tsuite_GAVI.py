#!/usr/bin/env python
#
# PYLOM Testsuite
# Run the GAVI R branch on the cylinder dataset: randomized QR, a short fit of the
# R variational autoencoder, its postprocess and a save/load round trip.
#
# The randomized QR draws a random basis, and the random generator differs between
# the Python and the compiled builds, so only basis independent quantities of the
# QR are checked. The autoencoder is trained on the R factor of the deterministic
# QR (positive diagonal) of the same data, which is the same in both builds.
#
# Last revision: 25/11/2026
import sys, os, numpy as np, torch
import pyLOM, pyLOM.NN

DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
SEED      = 42

pyLOM.NN.set_seed(SEED)
device = pyLOM.NN.select_device('cpu')
RESUDIR = os.path.join(OUTDIR,'GAVI_%s' % os.path.basename(DATAFILE).split('.')[0])
pyLOM.NN.create_results_folder(RESUDIR,verbose=False)


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
X = [d[v] for v in VARIABLES]
N = X[0].shape[1]


## Randomized QR keeping all the snapshots: B has the singular values of the
## mean subtracted data and Q has orthonormal columns, whatever the random basis
for v, Xv in zip(VARIABLES, X):
	Q, B = pyLOM.NN.GAVI.QR(Xv, N, q=1, osampl=8)
	sB   = np.linalg.svd(B, compute_uv=False)
	sX   = np.linalg.svd(Xv - Xv.mean(axis=1, keepdims=True), compute_uv=False)
	pyLOM.pprint(0,'TSUITE QR_%-9s =' % v,*B.shape,sB.max(),sB.sum(),np.abs(sB-sX).max()/sX.max(),np.abs(Q.T@Q-np.eye(N)).max())


## R factor of the deterministic QR, with a positive diagonal
def r_factor(Xv):
	R = np.linalg.qr(Xv - Xv.mean(axis=1, keepdims=True), mode='r')
	s = np.where(np.diag(R) < 0, -1., 1.)
	return s[:,np.newaxis]*R
R = tuple(r_factor(Xv) for Xv in X)


## R variational autoencoder: short fit and postprocess
latent_dim = 4
data, scaler = pyLOM.NN.GAVI.create_dataset(R, scale='max', device=device)
vae    = pyLOM.NN.GAVI.vae_R(data, latent_dim, nepochs=300, conv_chan=8, hid_dim=8, beta_first=100, beta_last=200, seed=SEED, BASEDIR=RESUDIR)
rectra = vae.reconstruct(data)
_,detR = vae.correlation(data)
energy = [pyLOM.NN.GAVI.energy(data,rectra,ichan) for ichan in range(len(VARIABLES))]
pyLOM.pprint(0,'TSUITE scaler       =',float(scaler))
pyLOM.pprint(0,'TSUITE rec          =',rectra.min(),rectra.max(),rectra.mean())
pyLOM.pprint(0,'TSUITE energy       =',*energy)
pyLOM.pprint(0,'TSUITE detR         =',detR)


## Save and load the autoencoder, the loaded one decodes the same latent vectors
latent = vae.latent_space(data)
pyLOM.NN.GAVI.save_vae_R(vae, BASEDIR=RESUDIR)
vae2   = pyLOM.NN.GAVI.load_vae_R(latent_dim, BASEDIR=RESUDIR)
dec    = vae.decode(latent)
dec2   = vae2.decode(latent)
pyLOM.pprint(0,'TSUITE latent       =',*latent.shape,latent.cpu().numpy().min(),latent.cpu().numpy().max())
pyLOM.pprint(0,'TSUITE load         =',np.abs(dec2-dec).max(),*dec.shape)


pyLOM.cr_info()
pyLOM.pprint(0,'End of output')
