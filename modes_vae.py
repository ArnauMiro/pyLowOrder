import torch
import pyLOM
import matplotlib.pyplot as plt
import numpy as np

## Specify autoencoder parameters
channels    = 32
lat_dim     = 5
beta        = 1e-3
kernel_size = 4
padding     = 1

## Load pyLOM dataset and set up results output
BASEDIR = '/home/benet/Dropbox/UNIVERSITAT/PhD/windsor/test_autoencoder'
CASESTR = 'back_dataset'
VARLIST = ['Cp']
DSETDIR = '%s/%s.h5' % (BASEDIR, CASESTR)
RESUSTR = 'vae_beta_%.2e_ld_%i/model_state' % (beta, lat_dim)

## Mesh size (HARDCODED BUT MUST BE INCLUDED IN PYLOM DATASET)
nx = 192
ny = 128

## Create a torch dataset
pyldtset = pyLOM.Dataset.load(DSETDIR)
tordtset = pyLOM.VAE.Dataset(pyldtset['Cp'], nx, ny, pyldtset.time)
loader   = torch.utils.data.DataLoader(tordtset, batch_size=len(tordtset), shuffle=False)
instant  = iter(loader)
batch    = next(instant)

encarch    = pyLOM.VAE.EncoderNoPool(lat_dim, nx, ny, channels, kernel_size, padding)
decarch    = pyLOM.VAE.DecoderNoPool(lat_dim, nx, ny, channels, kernel_size, padding)
vae        = pyLOM.VAE.VariationalAutoencoder(lat_dim, nx, ny, encarch, decarch)
vae.load_state_dict(torch.load(RESUSTR))

_, _, _, z = vae(batch)
ztest = z
ztest[:,1:] = 0
x_rec  = vae.decoder(ztest)
rec    = np.zeros((nx*ny,len(tordtset))) 
for i in range(len(tordtset)):
    x_recon  = x_rec[i,0,:,:]
    x_recon  = torch.reshape(torch.tensor(x_recon),[nx*ny, 1])
    rec[:,i] = x_recon.detach().numpy()[:,0]

##Save snapshots to paraview
visdtset = pyLOM.Dataset(ptable=pyldtset.partition_table, mesh=pyldtset.mesh, time=pyldtset.time)
visdtset.add_variable('Modes',True,1,rec)
visdtset.write('flow',basedir='flow',instants=np.arange(visdtset.time.shape[0],dtype=np.int32),times=visdtset.time,vars=['Modes'],fmt='vtkh5')

plt.figure()
for iz in range(lat_dim):
    plt.plot(ztest[:,iz].detach().numpy(), label='%i'%iz)
plt.legend()
plt.show()