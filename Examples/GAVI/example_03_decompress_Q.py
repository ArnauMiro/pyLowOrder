import torch
import numpy as np
import pyLOM, pyLOM.NN

## Input files
fmesh = '../pylom_mesh.h5'
frx   = 'QR_velox.h5'
fry   = 'QR_veloy.h5'
fcomp = 'compressed_data_array.h5'

## Load mesh
pOrder  = 4 ## pOrder of original SOD2D mesh for element grouping
nelxAE  = pOrder**3
mesh    = pyLOM.Mesh.load(fmesh)

## Load R
Rx   = pyLOM.NN.GAVI.load(frx, vars=['B'])[0]
Ry   = pyLOM.NN.GAVI.load(fry, vars=['B'])[0]
nmod = Rx.shape[0]

## Load compressed data (will be already on the needed DEVICE)
Qmeans, Qstds, weights, biases, Qs, Bs = pyLOM.NN.GAVI.load_compressed(fcomp, mesh.partition_table, nelxAE=nelxAE)

## Inference and reconstruct Qx to do velox
Qx = pyLOM.NN.GAVI.reconstruct_Q(mesh, nelxAE, nmod, Qmeans, Qstds, weights, biases, Qs, Bs, ivar=0)
velox = pyLOM.math.matmul(Qx, Rx[:,0])
del Qx
pyLOM.pprint(0, 'velox is reconstructed', flush=True)

## Inference and reconstruct Qy to do veloy
Qy = pyLOM.NN.GAVI.reconstruct_Q(mesh, nelxAE, nmod, Qmeans, Qstds, weights, biases, Qs, Bs, ivar=1)
veloy = pyLOM.math.matmul(Qy, Ry[:,0])
del Qy
pyLOM.pprint(0, 'veloy is reconstructed', flush=True)

## Build the dataset and write to paraview
d     = pyLOM.Dataset(xyz=mesh.xyz, ptable=mesh.partition_table, order=mesh.pointOrder, point=True, vars={'time':{'idim':0,'value':np.array([0.])}}, VELOX={'ndim':1,'value':velox[:,np.newaxis]}, VELOY={'ndim':1,'value':veloy[:,np.newaxis]})
pyLOM.io.pv_writer(mesh,d.to_cpu(['VELOX','VELOY']),'flow',basedir='out/flow',instants=np.array([0]),times=np.array([0.]),vars=['VELOX','VELOY'],fmt='vtkh5')

pyLOM.cr_info()