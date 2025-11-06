import torch
import numpy as np
import pyLOM
import matplotlib.pyplot as plt

# Parameters
FILE       = 'QR.h5'
latent_dim = 6

# Load data
#R = pyLOM.GAVI.load(FILE, vars=['B'])[0][:1600,:]
R = np.load('../gavi/POD/R.npy')

data,_ = pyLOM.GAVI.create_dataset(R, scale='max')
vae    = pyLOM.GAVI.vae_R(data, latent_dim)
rectra = vae.reconstruct(data)
_,detR = vae.correlation(data)
latent = vae.latent_space(data)

for ilat in range(latent.shape[1]):
    plt.figure(figsize=(8,6))
    plt.plot(latent[:,ilat].cpu().numpy(), 'r')
    plt.savefig('latents_%i.png' % ilat)


pyLOM.cr_info()