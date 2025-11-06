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

import torch
import numpy as np

from   ..NN import Encoder1D, Decoder1D, VariationalAutoencoder, silu, select_device, betaLinearScheduler


## Compute the randomized QR factorization

## Compress the randomized QR factorization

## Autoencoder on the R
def vae_R(data, latent_dim, device=select_device(), nepochs=2500, nlayers=3, conv_chan=64, hid_dim=32, kernel=4, padding=1, func=silu()):
    nmod       = data.shape[1]
    nt         = data.shape[0]
    input_chan = 1 if len(data.shape) == 2 else data.shape[2]
    activation = [func for _ in range(nlayers + 2)]
    encoder    = Encoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=True)
    decoder    = Decoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=True)
    vae        = VariationalAutoencoder(latent_dim, (nmod,), input_chan, encoder, decoder, device)
    vae.fit(data, eval_dataset=data, batch_size=64, epochs=nepochs, lr=5e-4, BASEDIR='./', pin_memory=False)
    return vae