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
beta        = 0.008
beta_start  = 0.008
beta_wmup   = 0
hidden_layer_sizes_enc = [1024, 512, 256, 128, 64]
hidden_layer_sizes_dec = [64, 128, 256, 512, 1024]
lat_dim                = 2
activations            = [pyLOM.NN.elu()]*len(hidden_layer_sizes_enc)


## Load pyLOM dataset and set up results output
RESUDIR = os.path.join(OUTDIR,'vae_beta_%.2e_ld_%i' % (beta, lat_dim))
pyLOM.NN.create_results_folder(RESUDIR,verbose=False)


## Load pyLOM dataset
m    = pyLOM.Mesh.load(DATAFILE)
d    = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
u    = d.X(*VARIABLES)
um   = pyLOM.math.temporal_mean(u)
u_x  = pyLOM.math.subtract_mean(u, um)
time = d.get_variable('time')


## Mesh size
n0h = 449
n0w = 199
in_size = n0w*n0h


## Create a torch dataset
u    = d.X(*VARIABLES)
um   = pyLOM.math.temporal_mean(u)
u_x  = pyLOM.math.subtract_mean(u, um).T
td   = pyLOM.NN.Dataset((u_x,), (in_size,))  # Flat the data


## Set and train the variational autoencoder
betasch    = pyLOM.NN.betaLinearScheduler(0., beta, beta_start, beta_wmup)
encoder    = pyLOM.NN.FullyConnectedEncoder2D(hidden_layer_sizes=hidden_layer_sizes_enc, lat_dim=lat_dim, in_size=in_size, activation_funcs=activations, vae=True)
decoder    = pyLOM.NN.FullyConnectedDecoder2D(hidden_layer_sizes=hidden_layer_sizes_dec, lat_dim=lat_dim, out_size=in_size, activation_funcs=activations)
model      = pyLOM.NN.FullyConnectedVariationalAutoencoder(latent_dim=lat_dim, in_size=in_size, encoder=encoder, decoder=decoder, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)

pipeline = pyLOM.NN.Pipeline(
    train_dataset = td,
    test_dataset  = td,
    model=model,
    training_params={
        "batch_size": 4,
        "epochs": 10,
        "lr": 1e-4,
        "betasch": betasch,
        "BASEDIR":RESUDIR
    },
)
pipeline.run()


## Reconstruct dataset and compute accuracy
rec = model.reconstruct(td)
rd  = pyLOM.NN.Dataset((rec,), (in_size,))
d.add_field('urec',1,rd[0].numpy().reshape((n0w*n0h,)))
d.add_field('utra',1,td[0].numpy().reshape((n0w*n0h,)))
pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=[0],times=[0.],vars=VARIABLES+['urec', 'utra'],fmt='vtkh5')


## Testsuite output
pyLOM.pprint(0,'TSUITE u_x  =',u_x.min(),u_x.max(),u_x.mean())
#pyLOM.pprint(0,'TSUITE urec =',d['urec'].min(),d['urec'].max(),d['urec'].mean())
pyLOM.pprint(0,'TSUITE utra =',d['utra'].min(),d['utra'].max(),d['utra'].mean())


pyLOM.cr_info()
pyLOM.pprint(0,'End of output')
