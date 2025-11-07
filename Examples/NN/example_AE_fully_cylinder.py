#!/usr/bin/env python
#
# Example of 2D-fully connected AE.
#
# Last revision: 07/11/2025

import numpy as np
import pyLOM, pyLOM.NN

import matplotlib.pyplot as plt


## Set device
device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Specify autoencoder parameters
hidden_layer_sizes_enc = [1024, 512, 256, 128, 64]
hidden_layer_sizes_dec = [64, 128, 256, 512, 1024]
lat_dim                = 2
activations            = [pyLOM.NN.elu()]*len(hidden_layer_sizes_enc)


## Load pyLOM dataset and set up results output
BASEDIR  = './DATA'
CASESTR = 'CYLINDER'
DSETDIR = '%s/%s.h5' % (BASEDIR, CASESTR)
RESUDIR = 'ae_ld_%i' % (lat_dim)
pyLOM.NN.create_results_folder(RESUDIR)


## Mesh size
n0h = 449
n0w = 199
in_size = n0w*n0h


## Create a torch dataset
m    = pyLOM.Mesh.load(DSETDIR)
d    = pyLOM.Dataset.load(DSETDIR,ptable=m.partition_table)
u    = d['VELOX']
um   = pyLOM.math.temporal_mean(u)
u_x  = pyLOM.math.subtract_mean(u, um).T

td   = pyLOM.NN.Dataset((u_x,), (in_size,))  # Flat the data

## Set and train the variational autoencoder
encoder    = pyLOM.NN.Encoder2Dfc(hidden_layer_sizes=hidden_layer_sizes_enc, lat_dim=lat_dim, in_size=in_size, activation_funcs=activations)
decoder    = pyLOM.NN.Decoder2Dfc(hidden_layer_sizes=hidden_layer_sizes_dec, lat_dim=lat_dim, out_size=in_size, activation_funcs=activations)
model      = pyLOM.NN.AutoencoderFully(latent_dim=lat_dim, in_size=in_size, encoder=encoder, decoder=decoder, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)

pipeline = pyLOM.NN.Pipeline(
   train_dataset   = td,
   test_dataset    = td,
   model           = model,
   training_params = {
       "batch_size": 16,
       "epochs": 100,
       "lr": 1e-4,
       "callback":early_stop,
       'BASEDIR':RESUDIR,
   },
   
)
pipeline.run()


## Reconstruct dataset and compute accuracy
rec = model.reconstruct(td)

rd  = pyLOM.NN.Dataset((rec,), (in_size,))

d.add_field('urec',1,rd[0].numpy().reshape((n0w*n0h,)))
d.add_field('utra',1,td[0].numpy().reshape((n0w*n0h,)))
pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=[0],times=[0.],vars=['urec', 'VELOX', 'utra'],fmt='vtkh5')
pyLOM.NN.plotSnapshot(m,d,vars=['urec'],instant=0,component=0,cmap='jet',cpos='xy')
pyLOM.NN.plotSnapshot(m,d,vars=['utra'],instant=0,component=0,cmap='jet',cpos='xy')

pyLOM.cr_info()
