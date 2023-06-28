import torch
import pyLOM
import numpy as np

## Load pyLOM tordtset
BASEDIR = '/home/benet/Dropbox/UNIVERSITAT/PhD/windsor/test_autoencoder'
CASESTR = 'back_dataset'
VARLIST = ['Cp']
DSETDIR = '%s/%s.h5' % (BASEDIR, CASESTR)

## Mesh size (HARDCODED BUT MUST BE INCLUDED IN PYLOM DATASET)
nx = 192
ny = 128

## Specify autoencoder parameters
ptrain        = 0.8
pvali         = 0.2
batch_size    = 32
nepochs       = 200
channels      = 16
lat_dim       = 5
beta          = 1e-3
learning_rate = 3e-4 #Karpathy Constant
kernel_size   = 4
stride        = 2
padding       = 1
results_file  = 'vae_beta_%.2e_ld_%i' % (beta, lat_dim)

## Create a torch dataset
pyldtset = pyLOM.Dataset.load(DSETDIR)
tordtset = pyLOM.VAE.Dataset(pyldtset['Cp'], nx, ny, pyldtset.time)

## Split data between train, test and validation
trloader, valoader = tordtset.split(ptrain, pvali, batch_size)

## Set and train the variational autoencoder
vae           = pyLOM.VAE.VariationalAutoencoder(channels, lat_dim, tordtset.nx, tordtset.ny, kernel_size, stride, padding)
early_stopper = pyLOM.VAE.EarlyStopper(patience=5, min_delta=0.02)
kld, mse, val_loss, train_loss_avg = vae.train_model(trloader, valoader, beta, nepochs, callback=early_stopper, learning_rate=learning_rate)
    
## Reconstruct dataset and compute accuracy
rec, energy = vae.reconstruct(tordtset)
cp_rec      = tordtset.recover(rec)
cp          = tordtset.recover(tordtset.data)
print('Recovered energy %.2f' % (energy))

##Save snapshots to paraview
visdtset = pyLOM.Dataset(ptable=pyldtset.partition_table, mesh=pyldtset.mesh, time=pyldtset.time)
visdtset.add_variable('Cp_rec',True,1,cp_rec)
visdtset.add_variable('Cp',True,1,cp)
visdtset.write('flow',basedir='flow',instants=np.arange(visdtset.time.shape[0],dtype=np.int32),times=visdtset.time,vars=['Cp_rec','Cp'],fmt='vtkh5')

## Compute the modes, its correlation and save them
dec            = vae.decoder
vae_modes      = dec.modes()
corrcoef, detR = vae.correlation(tordtset)
print('Correlation between modes %.2f' % (detR))
visdtset.add_variable('Modes',True,lat_dim,vae_modes)
visdtset.write(results_file, vars = ['Modes'], fmt='vtkh5')

## Save parameters and training results
pyLOM.VAE.save(vae.state_dict(), results_file, kld, mse, val_loss, train_loss_avg, corrcoef)