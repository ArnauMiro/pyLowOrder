#!/usr/bin/env python
#
# PYLOM Testsuite 2D-VAE.
#
# Last revision: 24/09/2024
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import sys, os, numpy as np
import pyLOM

DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]


## Set device
device = pyLOM.NN.select_device('cpu')


## Specify autoencoder parameters
ptrain      = 0.8
pvali       = 0.2
batch_size  = 1
nepochs     = 100
nlayers     = 1
channels    = 32
lat_dim     = 10
beta        = 0
kernel_size = 4
nlinear     = 256
padding     = 1
activations = [pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh()]
batch_norm  = True


## Load pyLOM dataset and set up results output
RESUDIR = os.path.join(OUTDIR,'vae_beta_%.2e_ld_%i' % (beta, lat_dim))
pyLOM.NN.create_results_folder(RESUDIR,echo=False)


## Load pyLOM dataset
m    = pyLOM.Mesh.load(DATAFILE)
d    = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
u_x  = d.X(*VARIABLES)
time = d.get_variable('time')


## Mesh size
n0h = len(np.unique(pyLOM.utils.round(d.xyz[:,0],5)))
n0w = len(np.unique(pyLOM.utils.round(d.xyz[:,1],5)))
nh  = 448
nw  = 192


## Create a torch dataset
td   = pyLOM.NN.Dataset((u_x,), (n0h, n0w), time, transform=False, device=device)
td.data[0] = np.transpose(np.array([td.data[0][:,0]]))
td._time   = np.array([td.time[0]])
td.crop((nh, nw), (n0h, n0w))
trloader = td.loader()


## Set and train the variational autoencoder
encarch    = pyLOM.NN.Encoder2D(nlayers, lat_dim, nh, nw, td.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm)
decarch    = pyLOM.NN.Decoder2D(nlayers, lat_dim, nh, nw, td.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm)
ae         = pyLOM.NN.Autoencoder(lat_dim, (nh, nw), td.n_channels, encarch, decarch, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)
ae.train_model(trloader, trloader, nepochs, callback=early_stop, BASEDIR=RESUDIR)


## Reconstruct dataset and compute accuracy
rec = ae.reconstruct(td)
rd  = pyLOM.NN.Dataset((rec), (nh, nw), td._time, transform=False)
rd.pad((nh, nw), (n0h, n0w))
td.pad((nh, nw), (n0h, n0w))
d.add_field('urec',1,rd.data[0][:,0].numpy())
d.add_field('utra',1,td.data[0][:,0].numpy())
pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=VARIABLES+['urec', 'utra'],fmt='vtkh5')


## Testsuite output
pyLOM.pprint(0,'TSUITE u_x  =',u_x.min(),u_x.max(),u_x.mean())
#pyLOM.pprint(0,'TSUITE urec =',d['urec'].min(),d['urec'].max(),d['urec'].mean())
pyLOM.pprint(0,'TSUITE utra =',d['utra'].min(),d['utra'].max(),d['utra'].mean())


pyLOM.cr_info()
pyLOM.pprint(0,'End of output')