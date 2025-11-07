#!/usr/bin/env python
#
# Example of 2D-fully connected VAE.
#
# Last revision: 07/11/2025
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import os, numpy as np
import pyLOM, pyLOM.NN


## Set device
device = pyLOM.NN.select_device("cuda") # Force CPU for this example, if left in blank it will automatically select the device


## Specify autoencoder parameters
beta        = 0.008
beta_start  = 0.008
beta_wmup   = 0
hidden_layer_sizes_enc = [1024, 512, 256, 128, 64]
hidden_layer_sizes_dec = [64, 128, 256, 512, 1024]
lat_dim                = 2
activations            = [pyLOM.NN.elu()]*len(hidden_layer_sizes_enc)


## Load pyLOM dataset and set up results output
BASEDIR = './DATA/'
CASESTR = 'CYLINDER'
DSETDIR = os.path.join(BASEDIR,f'{CASESTR}.h5')
RESUDIR = 'vae_beta_%.2e_ld_%i' % (beta, lat_dim)
pyLOM.NN.create_results_folder(RESUDIR)


## Mesh size (HARDCODED BUT MUST BE INCLUDED IN PYLOM DATASET)
n0h = 449
n0w = 199
in_size = n0w*n0h


## Create a torch dataset
m    = pyLOM.Mesh.load(DSETDIR)
d    = pyLOM.Dataset.load(DSETDIR,ptable=m.partition_table)
u_x  = d['VELOX']
u_m  = pyLOM.math.temporal_mean(u_x)
u_xm = pyLOM.math.subtract_mean(u_x, u_m).T
time = d.get_variable('time')
td   = pyLOM.NN.Dataset((u_xm,), (in_size,))


## Set and train the variational autoencoder
betasch    = pyLOM.NN.betaLinearScheduler(0., beta, beta_start, beta_wmup)
encoder    = pyLOM.NN.Encoder2Dfc(hidden_layer_sizes=hidden_layer_sizes_enc, lat_dim=lat_dim, in_size=in_size, activation_funcs=activations, vae=True)
decoder    = pyLOM.NN.Decoder2Dfc(hidden_layer_sizes=hidden_layer_sizes_dec, lat_dim=lat_dim, out_size=in_size, activation_funcs=activations)
model      = pyLOM.NN.VariationalAutoencoderFully(latent_dim=lat_dim, in_size=in_size, encoder=encoder, decoder=decoder, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=5, min_delta=0.02)

pipeline = pyLOM.NN.Pipeline(
    train_dataset = td,
    test_dataset  = td,
    model=model,
    training_params={
        "batch_size": 4,
        "epochs": 500,
        "lr": 1e-4,
        "betasch": betasch,
        "BASEDIR": RESUDIR
    },
)
pipeline.run()

## Reconstruct dataset and compute accuracy
rec = model.reconstruct(td)


## Fine-tuning process
RESUDIR_FT = f"{RESUDIR}/ft_vae_beta_{beta}_{lat_dim}"
pyLOM.NN.create_results_folder(RESUDIR_FT)

td_ft   = pyLOM.NN.Dataset((u_xm,), (in_size,))

z = model.latent_space(td_ft).cpu().numpy()
z_noisy = z + 10*np.random.rand(z.shape[0], z.shape[1], z.shape[2])  # Add noise to simulate regression prediction

dataset_train = np.concatenate((z_noisy, td_ft), axis=-1)
dataloader_params = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
        }

model.fine_tune(train_dataset=dataset_train, eval_dataset=dataset_train, epochs=100, shape_=td_ft.shape, BASEDIR=RESUDIR_FT, **dataloader_params)

ls = model.latent_space(td)
rec_ft = model.decode(ls)

print(np.abs(td[0].numpy()-rec[:, :, 0]).mean())
print(np.abs(td[0].numpy()-rec_ft[0]).mean())

breakpoint()
pyLOM.cr_info()