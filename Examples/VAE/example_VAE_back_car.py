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
lat_dim       = 20
beta          = 0
kernel_size   = 3
padding       = 1
results_file  = 'vae_beta_%.2e_ld_%i' % (beta, lat_dim)

## Create a torch dataset
pyldtset = pyLOM.Dataset.load(DSETDIR)
tordtset = pyLOM.VAE.Dataset(pyldtset['Cp'], nx, ny, pyldtset.time)

## Split data between train, test and validation
trloader, valoader = tordtset.split(ptrain, pvali, batch_size)

## Set and train the variational autoencoder
encarch    = pyLOM.VAE.EncoderMaxPool(lat_dim, nx, ny, channels, kernel_size, padding)
decarch    = pyLOM.VAE.DecoderMaxPool(lat_dim, nx, ny, channels, kernel_size, padding)
vae        = pyLOM.VAE.VariationalAutoencoder(lat_dim, nx, ny, encarch, decarch)
early_stop = pyLOM.VAE.EarlyStopper(patience=5, min_delta=0.02)
kld, mse, val_loss, train_loss_avg = vae.train_model(trloader, valoader, beta, nepochs, callback=early_stop)
    
## Reconstruct dataset and compute accuracy
rec    = vae.reconstruct(tordtset)
cp_rec = tordtset.recover(rec)
cp     = tordtset.recover(tordtset.data)

##Save snapshots to paraview
visdtset = pyLOM.Dataset(ptable=pyldtset.partition_table, mesh=pyldtset.mesh, time=pyldtset.time)
visdtset.add_variable('Cp_rec',True,1,cp_rec)
visdtset.add_variable('Cp',True,1,cp)
visdtset.write('flow',basedir='flow',instants=np.arange(visdtset.time.shape[0],dtype=np.int32),times=visdtset.time,vars=['Cp_rec','Cp'],fmt='vtkh5')

## Compute the modes, its correlation and save them
modes = vae.modes()
corr  = vae.correlation(tordtset)
visdtset.add_variable('Modes',True,lat_dim,modes)
visdtset.write(results_file, vars = ['Modes'], fmt='vtkh5')

## Save parameters and training results
pyLOM.VAE.save(vae.state_dict(), results_file, kld, mse, val_loss, train_loss_avg, corr)