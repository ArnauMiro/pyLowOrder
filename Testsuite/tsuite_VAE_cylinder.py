import pyLOM
import numpy as np
import matplotlib.pyplot as plt

## Set device
device = pyLOM.NN.select_device()

## Specify autoencoder parameters
ptrain      = 0.8
pvali       = 0.2
batch_size  = 1
nepochs     = 300
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
BASEDIR = '.'
CASESTR = 'CYLINDER'
DSETDIR = '%s/%s.h5' % (BASEDIR, CASESTR)
RESUDIR = 'vae_beta_%.2e_ld_%i' % (beta, lat_dim)
pyLOM.NN.create_results_folder(RESUDIR)

## Mesh size (HARDCODED BUT MUST BE INCLUDED IN PYLOM DATASET)
n0x = 449
n0y = 199
nx  = 448
ny  = 192

## Create a torch dataset
pyldtset = pyLOM.Dataset.load(DSETDIR)
u_x      = pyldtset['VELOX']
time     = pyldtset.time
tordtset = pyLOM.NN.Dataset((u_x,), n0x, n0y, time, transform=False)
tordtset.data[0] = np.transpose(np.array([tordtset.data[0][:,0]]))
tordtset._time   = np.array([tordtset.time[0]])
tordtset.crop(nx, ny, n0x, n0y)
trloader = tordtset.loader()

## Set and train the variational autoencoder
encarch    = pyLOM.NN.Encoder2D(nlayers, lat_dim, nx, ny, tordtset.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm)
decarch    = pyLOM.NN.Decoder2D(nlayers, lat_dim, nx, ny, tordtset.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm)
ae         = pyLOM.NN.Autoencoder(lat_dim, nx, ny, tordtset.n_channels, encarch, decarch, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)
ae.train_model(trloader, trloader, nepochs, callback=early_stop, BASEDIR=RESUDIR)
    
## Reconstruct dataset and compute accuracy
rec      = ae.reconstruct(tordtset)
recdtset = pyLOM.NN.Dataset((rec), nx, ny, tordtset._time, transform=False)
recdtset.pad(nx, ny, n0x, n0y)
tordtset.pad(nx, ny, n0x, n0y)
pyldtset.add_variable('urec', False, 1, recdtset.data[0][:,0].numpy())
pyldtset.add_variable('utra', False, 1, tordtset.data[0][:,0])
pyldtset.write('reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['urec', 'VELOX', 'utra'],fmt='vtkh5')