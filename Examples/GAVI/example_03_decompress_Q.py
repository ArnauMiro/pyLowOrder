import torch
import numpy as np
import pyLOM, pyLOM.NN

## Input files
fmesh = '../pylom_mesh.h5'
fr    = 'QR.h5'
fcomp = 'compressed_data_array.h5'

## Load mesh
pOrder  = 4 ## pOrder of original SOD2D mesh for element grouping
nelxAE  = pOrder**3
mesh    = pyLOM.Mesh.load(fmesh)

## Load R
R    = pyLOM.NN.GAVI.load(fr, vars=['B'])[0]
nmod = R.shape[0]

## Load compressed data (will be already on the needed DEVICE)
Qmeans, Qstds, weights, biases, Qs, Bs = pyLOM.NN.GAVI.load_compressed(fcomp, mesh.partition_table, nelxAE=nelxAE)

## Inference and reconstruct Q
Q = pyLOM.NN.GAVI.reconstruct_Q(mesh, nelxAE, nmod, Qmeans, Qstds, weights, biases, Qs, Bs)

## Reconstruct and visualize the first snapshot
reco = pyLOM.math.matmul(Q, R[:,0])
d    = pyLOM.Dataset(xyz=mesh.xyz, ptable=mesh.partition_table, order=mesh.pointOrder, point=True, vars={'time':{'idim':0,'value':np.array([0.])}}, VELOX={'ndim':1,'value':reco[:,np.newaxis]})
pyLOM.io.pv_writer(mesh,d.to_cpu(['VELOX']),'flow',basedir='out/flow',instants=np.array([0]),times=np.array([0.]),vars=['VELOX'],fmt='vtkh5')


pyLOM.cr_info()