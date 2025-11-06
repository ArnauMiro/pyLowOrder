import torch
import pyLOM

# Parameters
FILE       = 'QR.h5'
latent_dim = 4

# Load data
R = pyLOM.GAVI.load(FILE, vars=['B'])[0][:1600,:]

data,_ = pyLOM.GAVI.create_dataset(R, scale='max')

vae    = pyLOM.GAVI.vae_R(data, latent_dim)
rectra = vae.reconstruct(data)
_,detR = vae.correlation(data)
latent = vae.latent_space(data)

pyLOM.cr_info()