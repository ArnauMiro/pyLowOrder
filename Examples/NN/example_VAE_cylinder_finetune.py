#!/usr/bin/env python
#
# Example of 2D-VAE.
#
# Last revision: 24/09/2024
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import os, numpy as np
import pyLOM, pyLOM.NN


## Set device
device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Specify autoencoder parameters
nlayers     = 5
channels    = 64
lat_dim     = 5
beta        = 0
beta_start  = 0
beta_wmup   = 0
kernel_size = 4
nlinear     = 512
padding     = 1
activations = [pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu()]


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
u_m  = pyLOM.math.temporal_mean(u_x)
u_xm = pyLOM.math.subtract_mean(u_x, u_m)
time = d.get_variable('time')
td   = pyLOM.NN.Dataset((u_xm,), (n0h, n0w))
td.crop(nh, nw)


## Set and train the variational autoencoder
betasch    = pyLOM.NN.betaLinearScheduler(0., beta, beta_start, beta_wmup)
encoder    = pyLOM.NN.Encoder2D(nlayers, lat_dim, nh, nw, td.num_channels, channels, kernel_size, padding, activations, nlinear, vae=True)
decoder    = pyLOM.NN.Decoder2D(nlayers, lat_dim, nh, nw, td.num_channels, channels, kernel_size, padding, activations, nlinear)
model      = pyLOM.NN.VariationalAutoencoder(lat_dim, (nh, nw), td.num_channels, encoder, decoder, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)

pipeline = pyLOM.NN.Pipeline(
    train_dataset = td,
    test_dataset  = td,
    model=model,
    training_params={
        "batch_size": 4,
        "epochs": 100, 
        "lr": 1e-4,
        "betasch": betasch,
        "BASEDIR": RESUDIR
    },
)

pipeline.run()


## Reconstruct dataset and compute accuracy
rec = model.reconstruct(td)
rd  = pyLOM.NN.Dataset((rec,), (nh, nw))
rd.pad(n0h, n0w)
td.pad(n0h, n0w)
d.add_field('urec',1,rd[:,0,:,:].numpy().reshape((len(time),n0w*n0h)).T)
d.add_field('utra',1,td[:,0,:,:].numpy().reshape((len(time),n0w*n0h)).T)
pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['urec', 'VELOX', 'utra'],fmt='vtkh5')


## Fine tuning
RESUDIR_FT = f"{RESUDIR}/ft_vae_beta_{beta}_{lat_dim}"
pyLOM.NN.create_results_folder(RESUDIR_FT)

td_ft   = pyLOM.NN.Dataset((u_xm,), (n0h, n0w))
td_ft.crop(nh, nw)
z = model.latent_space(td_ft).cpu().numpy()
z_noisy = z + 10*np.random.rand(z.shape[0], z.shape[1])
td_rs = np.reshape(td_ft, (td_ft.shape[0]*td_ft.shape[1], td_ft.shape[2]*td_ft.shape[3]))
dataset_train = np.column_stack((z_noisy, td_rs))
dataloader_params = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
        }

model.fine_tune(train_dataset=dataset_train, eval_dataset=dataset_train, epochs=50, shape_=td_ft.shape, BASEDIR=RESUDIR_FT, **dataloader_params)
rec_ft = model.reconstruct(td_ft)
rd_ft  = pyLOM.NN.Dataset((rec_ft,), (nh, nw))
rd_ft.pad(n0h, n0w)
td_ft.pad(n0h, n0w)
d.add_field('urec_ft',1,rd_ft[:,0,:,:].numpy().reshape((len(time),n0w*n0h)).T)
d.add_field('utra_ft',1,td_ft[:,0,:,:].numpy().reshape((len(time),n0w*n0h)).T)
pyLOM.io.pv_writer(m,d,'reco_ft',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['urec_ft', 'VELOX', 'utra_ft'],fmt='vtkh5')

pyLOM.cr_info()