#!/usr/bin/env python
#
# PYLOM Testsuite 3D-VAE.
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
device = pyLOM.NN.select_device()


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
pyLOM.NN.create_results_folder(RESUDIR)


## Load the dataset
m    = pyLOM.Mesh.load(DATAFILE) # Mesh size (100 x 40 x 64)
d    = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
u    = d.X(*VARIABLES)
um   = pyLOM.math.temporal_mean(u)
u    = pyLOM.math.subtract_mean(u, um)
time = d.get_variable('time') # 120 instants
print("Variables: ", d.varnames)
print("Information about the variable: ", d.info(VARIABLES[0]))
print("Number of points ", len(d))
print("Instants :", time.shape[0])

# Take x component only for testing
nvars    = d.info(VARIABLES[0])['ndim']
u_x      = np.zeros((len(d),time.shape[0]), dtype=float)
u_x[:,:] = u[0:nvars*len(d):nvars,:]
print("New variable: u_x", u_x.shape)

# Mesh Size
n0x = len(np.unique(pyLOM.utils.truncate(d.x,6))) 
n0y = len(np.unique(pyLOM.utils.truncate(d.y,6)))
n0z = len(np.unique(pyLOM.utils.truncate(d.z,6)))
nx  = 64
ny  = 64
nz  = 64

# Create the torch dataset
td = pyLOM.NN.Dataset((u_x,), (n0x, n0y, n0z), time, transform=False, device=device)
td.crop((nx, ny, nz), (n0x, n0y, n0z))
trloader, valoader = td.split_subdatasets(ptrain, pvali,batch_size=batch_size)
#trloader = td.loader()

##Set beta scheduler
betasch = pyLOM.NN.betaLinearScheduler(0., beta, beta_start, beta_wmup)

## Set and train the Autoencoder
encarch = pyLOM.NN.Encoder3D(nlayers, lat_dim, nx, ny, nz, td.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm, stride = 2, dropout = 0, vae = vae)
decarch = pyLOM.NN.Decoder3D(nlayers, lat_dim, nx, ny, nz, td.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm)
AutoEnc = pyLOM.NN.VariationalAutoencoder(lat_dim, (nx, ny, nz), td.n_channels, encarch, decarch, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=15, min_delta=0.05)
AutoEnc.train_model(trloader, valoader, betasch, nepochs, callback = None, BASEDIR = RESUDIR)


## Reconstruct dataset and compute accuracy
rec = AutoEnc.reconstruct(td) # Returns (input channels, nx*ny, time)
rd  = pyLOM.NN.Dataset((rec), (nx, ny, nz), td._time, transform=False)
rd.pad((nx, ny, nz), (n0x, n0y, n0z))
td.pad((nx, ny, nz), (n0x, n0y, n0z))
d.add_field('urec', len(VARIABLES), rd.data[0][:,:].numpy())
d.add_field('utra', len(VARIABLES), td.data[0][:,:])
pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['urec','VELOC','utra'],fmt='vtkh5')
pyLOM.NN.plotSnapshot(m,d,vars=['urec'],instant=0,component=0,cmap='jet')
pyLOM.NN.plotSnapshot(m,d,vars=['utra'],instant=0,component=0,cmap='jet')


## Testsuite output
pyLOM.pprint(0,'TSUITE u_x  =',u_x.min(),u_x.max(),u_x.mean())
pyLOM.pprint(0,'TSUITE urec =',d['urec'].min(),d['urec'].max(),d['urec'].mean())
pyLOM.pprint(0,'TSUITE utra =',d['utra'].min(),d['utra'].max(),d['utra'].mean())


pyLOM.cr_info()
pyLOM.pprint(0,'End of output')