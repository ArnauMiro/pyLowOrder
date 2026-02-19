#!/usr/bin/env python
#
# Example: load saved latent vectors and trained GAVI R-VAE weights, decode to
# reconstruct Rx and Ry, then evaluate recovered energy on X and Y.
#
# Eiximeno, B., A., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025).
# On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models.
# Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797
#
# Last revision: 18/02/2025
#
import os
import numpy as np
import torch
import pyLOM.NN

# Match training setup from example_02A_GAVI_R.py
BASEDIR    = 'gavi_reconstruct_R/'
latent_dim = 6

# Load original R data and create dataset
Rx = pyLOM.NN.GAVI.load('QR_velox.h5', vars=['B'])[0]
Ry = pyLOM.NN.GAVI.load('QR_veloy.h5', vars=['B'])[0]
data, _ = pyLOM.NN.GAVI.create_dataset((Rx, Ry), scale='max')

# Load trained VAE (same architecture as vae_R)
vae = pyLOM.NN.GAVI.load_vae_R(data, latent_dim, BASEDIR=BASEDIR)

# Load latent vectors
latent = np.load(os.path.join(BASEDIR, 'latent_%i.npy' % latent_dim))

# Decode (output is scaled, same as dataset.variables_out)
with torch.no_grad():
    z   = torch.tensor(latent, dtype=torch.float32, device=pyLOM.NN.DEVICE)
    dec = vae.decode(z)  # raw shape (num_samples, inp_chan, N)

# Recovered energy on X and Y
energy_x = pyLOM.NN.GAVI.energy(data, dec, 0)
energy_y = pyLOM.NN.GAVI.energy(data, dec, 1)
pyLOM.pprint(0, 'Recovered energy X: {:.2f}%'.format(energy_x * 100), flush=True)
pyLOM.pprint(0, 'Recovered energy Y: {:.2f}%'.format(energy_y * 100), flush=True)

pyLOM.cr_info()
