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
beta          = 0.1
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

vae.train()
prev_train_loss = 1e99
train_loss_avg  = [] #Build numpy array as nepoch*num_batches
val_loss        = [] #Build numpy array as nepoch*num_batches*vali_batches
mse             = [] #Build numpy array as nepoch*num_batches
kld             = [] #Build numpy array as nepoch*num_batches
for epoch in range(nepochs):
    mse.append(0)
    kld.append(0)
    train_loss_avg.append(0)
    num_batches = 0 
    learning_rate = learning_rate * 1/(1 + 0.001 * epoch)   #learnign rate scheduled 
    optimizer = torch.optim.Adam(vae.parameters(), lr= learning_rate)
    for batch in trloader:     
        recon, mu, logvar, _ = vae(batch)
        mse_i  = vae.lossfunc(batch, recon)
        bkld_i = vae.kld(mu,logvar)*beta
        loss   = mse_i - bkld_i
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_avg[-1] += loss.item()
        mse[-1] = vae.lossfunc(batch, recon).item()
        kld[-1] = vae.kld(mu,logvar).item()*beta
        num_batches += 1
    with torch.no_grad():
        val_batches = 0
        val_loss.append(0)
        for val_batch in valoader:
            val_recon, val_mu, val_logvar , _ = vae(val_batch)
            mse_i     = vae.lossfunc(val_batch, val_recon)
            bkld_i    = vae.kld(mu,logvar)*beta
            vali_loss = mse_i - bkld_i
            val_loss[-1] += vali_loss.item()
            val_batches += 1
        val_loss[-1] /= num_batches
        mse[-1]      /= num_batches
        kld[-1]      /= num_batches
        train_loss_avg[-1] /= num_batches
    if early_stopper.early_stop(val_loss[-1], prev_train_loss, train_loss_avg[-1]):
        print('Early Stopper Activated at epoch %i' %epoch)
        break
    prev_train_loss = train_loss_avg[-1]   
    print('Epoch [%d / %d] average training error: %.5e' % (epoch+1, nepochs, train_loss_avg[-1]))

train_loss_avg = np.array(train_loss_avg)
val_loss       = np.array(val_loss)
mse            = np.array(mse)
kld            = np.array(kld)
    
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