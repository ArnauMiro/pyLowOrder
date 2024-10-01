#!/usr/bin/env python
#
# Example of 2D-VAE.
#
# Last revision: 24/09/2024
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import os, numpy as np
import pyLOM


## Set device
device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Specify autoencoder parameters
ptrain      = 0.8
pvali       = 0.2
batch_size  = 1
nepochs     = 10
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
BASEDIR = './DATA/'
CASESTR = 'CYLINDER'
DSETDIR = os.path.join(BASEDIR,f'{CASESTR}.h5')
RESUDIR = 'vae_beta_%.2e_ld_%i' % (beta, lat_dim)
pyLOM.NN.create_results_folder(RESUDIR)


## Mesh size (HARDCODED BUT MUST BE INCLUDED IN PYLOM DATASET)
n0h = 449
n0w = 199
nh  = 448
nw  = 192


## Create a torch dataset
m    = pyLOM.Mesh.load(DSETDIR)
d    = pyLOM.Dataset.load(DSETDIR,ptable=m.partition_table)
u_x  = d['VELOX']
time = d.get_variable('time')
td   = pyLOM.NN.Dataset((u_x,), (n0h, n0w), time, transform=False)
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
d.add_field('utra',1,td.data[0][:,0])
pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['urec', 'VELOX', 'utra'],fmt='vtkh5')
pyLOM.NN.plotSnapshot(m,d,vars=['urec'],instant=0,component=0,cmap='jet',cpos='xy')
pyLOM.NN.plotSnapshot(m,d,vars=['utra'],instant=0,component=0,cmap='jet',cpos='xy')


pyLOM.cr_info()