#!/usr/bin/env python
#
# Example on how to compress the Q matrix resulting from a randomized QR factorization using GAVI.
#
# ADD CITATION AFTER PUBLICATION OF MADRID PROCEEDINGS
#
# Last revision: 13/11/2025
import torch
import pyLOM, pyLOM.NN

## Architecture Parameters
r          = 10  ## Number of modes of the latent vectors that we'll retain
porder     = 4   ## Input the pOrder of the original mesh to group the rest of elements
out_file   = 'compressed_data_array'

## Load data
mesh    = pyLOM.Mesh.load('../pylom_mesh.h5')
Qx      = pyLOM.NN.GAVI.load('QR_velox.h5', vars=['Q'], ptable=mesh.partition_table)[0]
Qy      = pyLOM.NN.GAVI.load('QR_veloy.h5', vars=['Q'], ptable=mesh.partition_table)[0]

## Compres the Q represented in the loaded mesh
pyLOM.NN.GAVI.vae_Q(out_file,(Qx,Qy),mesh,porder,r)

## Print timings
pyLOM.cr_info()