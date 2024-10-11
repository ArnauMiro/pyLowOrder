#!/usr/bin/env python
#
# Example of 3D-VAE.
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
batch_size  = 4
nepochs     = 10
nlayers     = 4
channels    = 48
lat_dim     = 10
beta        = 1e-04
beta_wmup   = 500
beta_start  = 30
kernel_size = 4
nlinear     = 256
padding     = 1
activations = [pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu()]
batch_norm  = False
vae         = True


## Load dataset and set up the results output
BASEDIR  = './DATA/'
CASESTR  = 'CHANNEL'
DSETDIR  = os.path.join(BASEDIR,f'{CASESTR}.h5')
VARIABLE = 'VELOX'
RESUDIR  = 'vae_beta_%.2e_ld_%i' % (beta, lat_dim)
pyLOM.NN.create_results_folder(RESUDIR)


## Load the dataset
m    = pyLOM.Mesh.load(DSETDIR) # Mesh size (100 x 40 x 64)
d    = pyLOM.Dataset.load(DSETDIR,ptable=m.partition_table)
u    = d[VARIABLE] # vars ['VELOC'] : u
um   = pyLOM.math.temporal_mean(u)
u_x  = pyLOM.math.subtract_mean(u, um)
time = d.get_variable('time') # 120 instants
pyLOM.pprint("Variables: ", d.varnames)
pyLOM.pprint("Information about the variable: ", d.info(VARIABLE))
pyLOM.pprint("Number of points ", len(d))
pyLOM.pprint("Instants :", time.shape[0])


## Mesh Size
n0x = len(np.unique(np.round(d.xyz[:,0],5)))
n0y = len(np.unique(np.round(d.xyz[:,1],5))) 
n0z = len(np.unique(np.round(d.xyz[:,2],5))) 
nx, ny, nz = 64, 64, 64


## Create the torch dataset
td = pyLOM.NN.Dataset((u_x,), (n0x, n0y, n0z))
td.crop(nx, ny, nz)


## Set and train the variational autoencoder
#betasch    = pyLOM.NN.betaLinearScheduler(0., beta, beta_start, beta_wmup)
#encoder    = pyLOM.NN.Encoder3D(nlayers, lat_dim, nx, ny, nz, td.num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm, stride = 2, dropout = 0, vae = vae)
#decoder    = pyLOM.NN.Decoder3D(nlayers, lat_dim, nx, ny, nz, td.num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm)
#model      = pyLOM.NN.VariationalAutoencoder(lat_dim, (nx, ny, nz), td.num_channels, encoder, decoder, device=device)
#early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)
#
#pipeline = pyLOM.NN.Pipeline(
#    train_dataset = td,
#    test_dataset  = td,
#    model=model,
#    training_params={
#        "batch_size": 1,
#        "epochs": 500,
#        "lr": 1e-4,
#        "betasch": betasch
#    },
#)
#pipeline.run()


## Reconstruct dataset and compute accuracy
#rec = model.reconstruct(td)
#rd  = pyLOM.NN.Dataset((rec,), (nx, ny, nz))
#rd.pad((nx, ny, nz), (n0x, n0y, n0z))
td.pad(nx, ny, nz)
#d.add_field('urec', 1, rd.data[0][:,:].numpy())
#d.add_field('utra',1,td[0,0,:,:,:].numpy().reshape((nx*ny*nz,)))
#pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['urec','VELOX','utra'],fmt='vtkh5')
#pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['VELOX','utra'],fmt='vtkh5')


pyLOM.cr_info()