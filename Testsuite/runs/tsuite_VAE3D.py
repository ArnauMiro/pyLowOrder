#!/usr/bin/env python
#
# PYLOM Testsuite 3D-VAE.
#
# Last revision: 24/09/2024
from __future__ import print_function, division
 
import mpi4py
mpi4py.rc.recv_mprobe = False

import sys, os, numpy as np, json
import pyLOM, pyLOM.NN

DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
PARAMS    = json.loads(str(sys.argv[4]).replace("'",'"'))


## Set device
device = pyLOM.NN.select_device("cpu")


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
RESUDIR  = os.path.join(OUTDIR,'vae_beta_%.2e_ld_%i' % (beta, lat_dim))
pyLOM.NN.create_results_folder(RESUDIR,verbose=False)


## Load the dataset
m    = pyLOM.Mesh.load(DATAFILE) # Mesh size (100 x 40 x 64)
d    = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
u    = d.X(*VARIABLES)
um   = pyLOM.math.temporal_mean(u)
u_x  = pyLOM.math.subtract_mean(u, um)
time = d.get_variable('time') # 120 instants

# Mesh Size
n0x = len(np.unique(pyLOM.utils.round(d.xyz[:,0],5)))
n0y = len(np.unique(pyLOM.utils.round(d.xyz[:,1],5))) 
n0z = len(np.unique(pyLOM.utils.round(d.xyz[:,2],5))) 
nx, ny, nz  = PARAMS['nx'], PARAMS['ny'], PARAMS['nz']

# Create the torch dataset
td = pyLOM.NN.Dataset((u_x,), (n0x, n0y, n0z))
td.crop(nx, ny, nz)


## Set and train the variational autoencoder
betasch    = pyLOM.NN.betaLinearScheduler(0., beta, beta_start, beta_wmup)
encoder    = pyLOM.NN.Encoder3D(nlayers, lat_dim, nx, ny, nz, td.num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm, stride = 2, dropout = 0, vae = vae)
decoder    = pyLOM.NN.Decoder3D(nlayers, lat_dim, nx, ny, nz, td.num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm)
model      = pyLOM.NN.VariationalAutoencoder(lat_dim, (nx, ny, nz), td.num_channels, encoder, decoder, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=15, min_delta=0.05)

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
rd  = pyLOM.NN.Dataset((rec,), (nx, ny, nz))
rd.pad(n0x, n0y, n0z)
td.pad(n0x, n0y, n0z)
d.add_field('urec',1,rd[:,0,:,:].numpy().reshape((len(time),n0x*n0y*n0z)).T)
d.add_field('utra',1,td[:,0,:,:].numpy().reshape((len(time),n0x*n0y*n0z)).T)
pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=VARIABLES+['urec','utra'],fmt='vtkh5')


## Testsuite output
pyLOM.pprint(0,'TSUITE u_x  =',u.min(),u.max(),u.mean())
#pyLOM.pprint(0,'TSUITE urec =',d['urec'].min(),d['urec'].max(),d['urec'].mean())
pyLOM.pprint(0,'TSUITE utra =',d['utra'].min(),d['utra'].max(),d['utra'].mean())


pyLOM.cr_info()
pyLOM.pprint(0,'End of output')