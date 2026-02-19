#!/usr/bin/env python
#
# Example on how to run fit an autoencoder for the temporal dynamics of a system following the GAVI methodology.
#
# Eiximeno, B., A., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025). 
# On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models. 
# Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797
#
# Last revision: 07/11/2025
import pyLOM.NN


## Set device
device = pyLOM.NN.select_device() # Automatically select the device


# Architecture Parameters
latent_dim = 8

# Load data
BASEDIR = './'
Rx = pyLOM.NN.GAVI.load('QR_velox.h5', vars=['B'])[0]
Ry = pyLOM.NN.GAVI.load('QR_veloy.h5', vars=['B'])[0]

# Create NN dataset
data,_ = pyLOM.NN.GAVI.create_dataset((Rx,Ry), scale='max')

# Create and train the autoencoder
vae    = pyLOM.NN.GAVI.vae_R(data, latent_dim, nepochs=1000, BASEDIR=BASEDIR)

# Postprocess trained autoencoder
rectra = vae.reconstruct(data)
_,detR = vae.correlation(data)

# Compute energy of the recovered data for each variable
energyX = pyLOM.NN.GAVI.energy(data,rectra,0)
pyLOM.pprint(0,"Recovered energy X: {:.2f}%".format(energyX*100),flush=True)
energyY = pyLOM.NN.GAVI.energy(data,rectra,1)
pyLOM.pprint(0,"Recovered energy Y: {:.2f}%".format(energyY*100),flush=True)

# Project and save latent space
latent = vae.latent_space(data)
np.save(os.path.join(BASEDIR, "latent_%i.npy" % latent_dim), latent.cpu().numpy())

pyLOM.cr_info()
