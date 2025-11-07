#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# GAVI (Geometry Agnostic Variational-autoencoders Integration) interface.
#
# Eiximeno, B., A., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025). 
# On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models. 
# Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797
#
# Last rev: 24/10/2025

from ..      import Encoder1D, Decoder1D, VariationalAutoencoder, betaLinearScheduler
from ..      import silu, select_device
from ..      import temporal_mean, subtract_mean, randomized_qr2
from ..utils import cr


## Compute the randomized QR factorization
@cr('GAVI.QR')
def QR(X, k, q=1, osampl=10):
    r   = k+osampl if k+osampl < X.shape[1] else X.shape[1]
    Xm  = temporal_mean(X)
    X   = subtract_mean(X, Xm)
    Q,B = randomized_qr2(X,r,q)
    
    return Q[:,:k].copy(), B[:k,:].copy()

## Compress the randomized QR factorization

## Autoencoder on the R
def vae_R(data, latent_dim, device=select_device(), nepochs=2500, nlayers=3, conv_chan=64, hid_dim=32, kernel=4, padding=1, func=silu()):
    nmod       = data.shape[2]
    input_chan = data.shape[1]
    activation = [func for _ in range(nlayers + 2)]
    encoder    = Encoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
    decoder    = Decoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
    vae        = VariationalAutoencoder(latent_dim, (nmod,), input_chan, encoder, decoder, device)
    vae.fit(data, eval_dataset=data, betasch=betaLinearScheduler(0,2.5e-2,500,1000), batch_size=64, epochs=nepochs, lr=7.5e-4, BASEDIR='./', pin_memory=False)
    return vae