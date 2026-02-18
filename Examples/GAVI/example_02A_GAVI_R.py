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


## Set device
device = pyLOM.NN.select_device() # Automatically select the device


# Parameters
latent_dim = 8

# Load data
Rx = pyLOM.NN.GAVI.load('QR_velox.h5', vars=['B'])[0]
Ry = pyLOM.NN.GAVI.load('QR_veloy.h5', vars=['B'])[0]

data,_ = pyLOM.NN.GAVI.create_dataset((Rx,Ry), scale='max')

vae    = pyLOM.NN.GAVI.vae_R(data, latent_dim, nepochs=1000)
rectra = vae.reconstruct(data)
_,detR = vae.correlation(data)
latent = vae.latent_space(data)

energyX = pyLOM.NN.GAVI.energy(data,rectra,0)
pyLOM.pprint(0,"Recovered energy X: {:.2f}%".format(energyX*100),flush=True)
energyY = pyLOM.NN.GAVI.energy(data,rectra,1)
pyLOM.pprint(0,"Recovered energy Y: {:.2f}%".format(energyY*100),flush=True)

pyLOM.cr_info()
