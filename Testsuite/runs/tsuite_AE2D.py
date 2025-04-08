#!/usr/bin/env python
#
# PYLOM Testsuite 2D-VAE.
#
# Last revision: 24/09/2024
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import sys, os, numpy as np
import pyLOM, pyLOM.NN

DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]


## Set device
device = pyLOM.NN.select_device('cpu')


## Specify autoencoder parameters
nlayers     = 5
channels    = 64
lat_dim     = 5
kernel_size = 4
nlinear     = 512
padding     = 1
activations = [pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh()]


## Load pyLOM dataset and set up results output
RESUDIR = os.path.join(OUTDIR,'ae_ld_%i' % (lat_dim))
pyLOM.NN.create_results_folder(RESUDIR,verbose=False)


## Load pyLOM dataset
m    = pyLOM.Mesh.load(DATAFILE)
d    = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)


## Mesh size
n0h = len(np.unique(pyLOM.utils.round(d.xyz[:,0],5)))
n0w = len(np.unique(pyLOM.utils.round(d.xyz[:,1],5)))
nh  = 448
nw  = 192


## Create a torch dataset
u    = d.X(*VARIABLES)
um   = pyLOM.math.temporal_mean(u)
u_x  = pyLOM.math.subtract_mean(u, um)[:,0]
td   = pyLOM.NN.Dataset((u_x,), (n0h, n0w))
td.crop(nh, nw)


## Set and train the variational autoencoder
encoder    = pyLOM.NN.Encoder2D(nlayers, lat_dim, nh, nw, td.num_channels, channels, kernel_size, padding, activations, nlinear)
decoder    = pyLOM.NN.Decoder2D(nlayers, lat_dim, nh, nw, td.num_channels, channels, kernel_size, padding, activations, nlinear)
model      = pyLOM.NN.Autoencoder(lat_dim, (nh, nw), td.num_channels, encoder, decoder, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)

pipeline = pyLOM.NN.Pipeline(
   train_dataset   = td,
   test_dataset    = td,
   model           = model,
   training_params = {
       "batch_size": 16,
       "epochs": 10,
       "lr": 1e-4,
       "callback":early_stop,
       'BASEDIR':RESUDIR,
   },
   
)
pipeline.run()


## Reconstruct dataset and compute accuracy
rec = model.reconstruct(td)
rd  = pyLOM.NN.Dataset((rec), (nh, nw))
rd.pad(n0h,n0w)
td.pad(n0h,n0w)
d.add_field('urec',1,rd[0,0,:,:].numpy().reshape((n0w*n0h,)))
d.add_field('utra',1,td[0,0,:,:].numpy().reshape((n0w*n0h,)))
pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=[0],times=[0.],vars=VARIABLES+['urec', 'utra'],fmt='vtkh5')


## Testsuite output
pyLOM.pprint(0,'TSUITE u_x  =',u_x.min(),u_x.max(),u_x.mean())
#pyLOM.pprint(0,'TSUITE urec =',d['urec'].min(),d['urec'].max(),d['urec'].mean())
pyLOM.pprint(0,'TSUITE utra =',d['utra'].min(),d['utra'].max(),d['utra'].mean())


pyLOM.cr_info()
pyLOM.pprint(0,'End of output')