#!/usr/bin/env python
#
# Example on how to run fit an autoencoder for the temporal dynamics of a system following the GAVI methodology.
#
# Eiximeno, B., A., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025). 
# On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models. 
# Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797
#
# Last revision: 07/11/2025
import torch
import pyLOM.NN

# Parameters
FILE       = 'QR.h5'
latent_dim = 4

# Load data
R = pyLOM.NN.GAVI.load(FILE, vars=['B'])[0]

data,_ = pyLOM.NN.GAVI.create_dataset(R, scale='max')

vae    = pyLOM.NN.GAVI.vae_R(data, latent_dim)
rectra = vae.reconstruct(data)
_,detR = vae.correlation(data)
latent = vae.latent_space(data)

pyLOM.cr_info()