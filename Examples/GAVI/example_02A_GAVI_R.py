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
latent_dim = 8

# Load data
Rx = pyLOM.NN.GAVI.load('QR_velox.h5', vars=['B'])[0]
Ry = pyLOM.NN.GAVI.load('QR_veloy.h5', vars=['B'])[0]
R  = torch.zeros((Rx.shape[0],2,Rx.shape[1]), dtype=torch.float32, device=pyLOM.NN.DEVICE)
R[:,0,:] = torch.tensor(Rx, dtype=torch.float32, device=pyLOM.NN.DEVICE)
R[:,1,:] = torch.tensor(Ry, dtype=torch.float32, device=pyLOM.NN.DEVICE)

data,_ = pyLOM.NN.GAVI.create_dataset(R, scale='max')

vae    = pyLOM.NN.GAVI.vae_R(data, latent_dim, nepochs=1000)
rectra = vae.reconstruct(data)
_,detR = vae.correlation(data)
latent = vae.latent_space(data)

energyX = pyLOM.math.energy(data.variables_out[:,0,:].cpu().numpy().T, rectra[0])
print("Recovered energy X: {:.2f}%".format(energyX*100))
energyY = pyLOM.math.energy(data.variables_out[:,1,:].cpu().numpy().T, rectra[1])
print("Recovered energy Y: {:.2f}%".format(energyY*100))

pyLOM.cr_info()
