import pyLOM
import numpy as np
import matplotlib.pyplot as plt

## Set device
device = pyLOM.NN.select_device()

## Specify autoencoder parameters
nlayers     = 5
channels    = 64
lat_dim     = 5
beta        = 0
kernel_size = 4
nlinear     = 512
padding     = 1
activations = [pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh(), pyLOM.NN.tanh()]

## Load pyLOM dataset and set up results output
BASEDIR = 'Testsuite'
CASESTR = 'CYLINDER'
DSETDIR = '%s/%s.h5' % (BASEDIR, CASESTR)
RESUDIR = 'vae_beta_%.2e_ld_%i' % (beta, lat_dim)
pyLOM.NN.create_results_folder(RESUDIR)

## Mesh size
n0h = 449
n0w = 199
nh  = 448
nw  = 192

## Create a torch dataset
pyldtset = pyLOM.Dataset.load(DSETDIR)
u_x      = pyldtset['VELOX'][:nh*nw,0]
tordtset = pyLOM.NN.Dataset((u_x[:,np.newaxis],), (nh, nw))
time     = np.array([0])

## Set and train the variational autoencoder
encoder    = pyLOM.NN.Encoder2D(nlayers, lat_dim, nh, nw, tordtset.num_channels, channels, kernel_size, padding, activations, nlinear)
decoder    = pyLOM.NN.Decoder2D(nlayers, lat_dim, nh, nw, tordtset.num_channels, channels, kernel_size, padding, activations, nlinear)
model      = pyLOM.NN.Autoencoder(lat_dim, (nh, nw), tordtset.num_channels, encoder, decoder, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)

pipeline = pyLOM.NN.Pipeline(
    train_dataset = tordtset,
    test_dataset = tordtset,
    model=model,
    training_params={
        "batch_size": 1,
        "epochs": 10000,
        "lr": 1e-4
    },
)
pipeline.run()

    
## Reconstruct dataset and compute accuracy
rec = model.reconstruct(tordtset)
recfile = np.zeros((n0h*n0w,1))
recfile[:nh*nw,:] = rec[0,:]
pyldtset.add_variable('VELOR', False, 1, recfile)
pyldtset.add_variable('utra', False, 1, u_x)
pyldtset.write('reco',basedir='.',instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['VELOX', 'VELOR'],fmt='vtkh5')
