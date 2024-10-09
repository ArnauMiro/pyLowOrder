#!/usr/bin/env python
#
# Example of 2D-AE.
#
# Last revision: 09/10/2024

import numpy as np
import pyLOM

import matplotlib.pyplot as plt


## Set device
device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Specify autoencoder parameters
nlayers     = 5
channels    = 64
lat_dim     = 5
kernel_size = 4
nlinear     = 512
padding     = 1
activations = [pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh()]


## Load pyLOM dataset and set up results output
BASEDIR  = './DATA/'
CASESTR = 'CYLINDER'
DSETDIR = '%s/%s.h5' % (BASEDIR, CASESTR)
RESUDIR = 'ae_ld_%i' % (lat_dim)
pyLOM.NN.create_results_folder(RESUDIR)


## Mesh size
n0h = 449
n0w = 199
nh  = 448
nw  = 192


## Create a torch dataset
m    = pyLOM.Mesh.load(DSETDIR)
d    = pyLOM.Dataset.load(DSETDIR,ptable=m.partition_table)
u_x  = d['VELOX']
td   = pyLOM.NN.Dataset((u_x[:,np.newaxis],), (nh, nw))
time = np.array([0])


## Set and train the variational autoencoder
encoder    = pyLOM.NN.Encoder2D(nlayers, lat_dim, nh, nw, td.num_channels, channels, kernel_size, padding, activations, nlinear)
decoder    = pyLOM.NN.Decoder2D(nlayers, lat_dim, nh, nw, td.num_channels, channels, kernel_size, padding, activations, nlinear)
model      = pyLOM.NN.Autoencoder(lat_dim, (nh, nw), td.num_channels, encoder, decoder, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)

pipeline = pyLOM.NN.Pipeline(
    train_dataset = td,
    test_dataset = td,
    model=model,
    training_params={
        "batch_size": 1,
        "epochs": 10000,
        "lr": 1e-4
    },
)
pipeline.run()

    
## Reconstruct dataset and compute accuracy
rec = model.reconstruct(td)
rd  = np.zeros((n0h*n0w,1))
rd[:nh*nw,:] = rec[0,:]
d.add_field('VELOR',1,rd)
d.add_field('utra',1,td.data[0][:,0].numpy())
pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['VELOX', 'VELOR', 'utra'],fmt='vtkh5')
pyLOM.NN.plotSnapshot(m,d,vars=['VELOR'],instant=0,component=0,cmap='jet',cpos='xy')
pyLOM.NN.plotSnapshot(m,d,vars=['utra'],instant=0,component=0,cmap='jet',cpos='xy')
