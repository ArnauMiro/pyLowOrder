import pyLOM
import numpy as np

## Specify autoencoder parameters
ptrain      = 0.8
pvali       = 0.2
batch_size  = 2
nepochs     = 200
channels    = 32
lat_dim     = 5
beta        = 1e-4
kernel_size = 4
padding     = 1

## Load pyLOM dataset and set up results output
BASEDIR = '/home/benet/Dropbox/UNIVERSITAT/PhD/windsor/test_autoencoder'
CASESTR = 'back_dataset'
VARLIST = ['Cp']
DSETDIR = '%s/%s.h5' % (BASEDIR, CASESTR)
RESUDIR = 'vae_beta_%.2e_ld_%i' % (beta, lat_dim)
pyLOM.VAE.create_results_folder(RESUDIR)

## Mesh size (HARDCODED BUT MUST BE INCLUDED IN PYLOM DATASET)
nx = 192
ny = 128

## Create a torch dataset
pyldtset = pyLOM.Dataset.load(DSETDIR)
tordtset = pyLOM.VAE.Dataset(pyldtset['Cp'], nx, ny, pyldtset.time)

## Split data between train, test and validation
trloader, valoader = tordtset.split(ptrain, pvali,batch_size=batch_size)

## Set and train the variational autoencoder
encarch    = pyLOM.VAE.EncoderNoPool(lat_dim, nx, ny, channels, kernel_size, padding)
decarch    = pyLOM.VAE.DecoderNoPool(lat_dim, nx, ny, channels, kernel_size, padding)
vae        = pyLOM.VAE.VariationalAutoencoder(lat_dim, nx, ny, encarch, decarch)
early_stop = pyLOM.VAE.EarlyStopper(patience=5, min_delta=0.02)
vae.train_model(trloader, valoader, beta, nepochs, callback=early_stop, BASEDIR=RESUDIR)
    
## Reconstruct dataset and compute accuracy
rec    = vae.reconstruct(tordtset)
cp_rec = tordtset.recover(rec)
cp     = tordtset.recover(tordtset.data)

##Save snapshots to paraview
visdtset = pyLOM.Dataset(ptable=pyldtset.partition_table, mesh=pyldtset.mesh, time=pyldtset.time)
visdtset.add_variable('Cp_rec',True,1,cp_rec)
visdtset.add_variable('Cp',True,1,cp)
visdtset.write('flow',basedir='%s/flow'%RESUDIR,instants=np.arange(visdtset.time.shape[0],dtype=np.int32),times=visdtset.time,vars=['Cp_rec','Cp'],fmt='vtkh5')

## Compute the modes, its correlation and save them
modes = vae.modes()
corr  = vae.correlation(tordtset)
visdtset.add_variable('Modes',True,lat_dim,modes)
visdtset.write('modes',basedir=RESUDIR, vars = ['Modes'], fmt='vtkh5')