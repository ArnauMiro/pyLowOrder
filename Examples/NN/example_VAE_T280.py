## 3D CNN Autoencoder Test 
## T280
## Aleix Usieda
 

import pyLOM
import numpy as np

## Set device

device = pyLOM.NN.select_device()

## Specify autoencoder parameters
ptrain      = 0.8
pvali       = 0.2
batch_size  = 4
nepochs     = 2000
nlayers     = 4
channels    = 48
lat_dim     = 10
beta        = 1e-04
kernel_size = 4
nlinear     = 256
padding     = 1
activations = [pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu()]
batch_norm  = False
vae         = True

## Load dataset and set up the results output

DATAFILE = 'Examples/Data/Tensor_re280.h5'
VARIABLE = 'VELOC'
RESUDIR = 'vae_beta_%.2e_ld_%i' % (beta, lat_dim)
pyLOM.NN.create_results_folder(RESUDIR)

## Mesh size (100 x 40 x 64)
## vars ['VELOC'] : u
## 120 instants

## Load the dataset
pyldtset = pyLOM.Dataset.load(DATAFILE)
u        = pyldtset[VARIABLE]
um       = pyLOM.math.temporal_mean(u)
u        = pyLOM.math.subtract_mean(u, um)
time     = pyldtset.time 
mesh     = pyldtset.mesh
print("Variables: ", pyldtset.varnames)
print("Information about the variable: ", pyldtset.info(VARIABLE))
print("Number of cells ", mesh.ncells)
print("Instants :", time.shape[0])

#Take x component only for testing
nvars = pyldtset._vardict[VARIABLE]['ndim']
u_x = np.zeros((mesh.ncells,time.shape[0]), dtype = float)
u_x[:,:] = u[0:nvars*mesh.ncells:nvars,:]
print("New variable: u_x", u_x.shape)

# Mesh Size
n0x = len(np.unique(mesh.x))-1 
n0y = len(np.unique(mesh.y))-1
n0z = len(np.unique(mesh.z))-1
nx = 96
ny = 32
nz = n0z


#Create the torch dataset
tordtset = pyLOM.NN.Dataset3D((u_x,), n0x, n0y, n0z, time, transform=False, device=device)

'''
#Single Snapshot
tordtset.data[0] = np.transpose(np.array([tordtset.data[0][:,0]]))
tordtset._time = np.array([tordtset.time[0]])
'''


tordtset.crop(nx, ny, nz, n0x, n0y, n0z)
trloader, valoader = tordtset.split_subdatasets(ptrain, pvali,batch_size=batch_size)
#trloader = tordtset.loader()


## Set and train the Autoencoder
encarch = pyLOM.NN.Encoder3D(nlayers, lat_dim, nx, ny, nz, tordtset.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm, stride = 2, dropout = 0, vae = vae)
decarch = pyLOM.NN.Decoder3D(nlayers, lat_dim, nx, ny, nz, tordtset.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm)
AutoEnc = pyLOM.NN.VariationalAutoencoder(lat_dim, (nx, ny, nz), tordtset.n_channels, encarch, decarch, device=device)
early_stop = pyLOM.NN.EarlyStopper(patience=15, min_delta=0.05)
AutoEnc.train_model(trloader, valoader, beta, nepochs, callback = None, BASEDIR = RESUDIR)
#AutoEnc.load_state_dict(torch.load(MODEL_PATH))


## Reconstruct dataset and compute accuracy
rec  = AutoEnc.reconstruct(tordtset) # Returns (input channels, nx*ny, time)
recdtset = pyLOM.NN.Dataset3D((rec), nx, ny, nz, tordtset._time, transform=False)
recdtset.pad(nx, ny, nz, n0x, n0y, n0z)
tordtset.pad(nx, ny, nz, n0x, n0y, n0z)
pyldtset.add_variable('urec', False, 1, recdtset.data[0][:,:].numpy())
pyldtset.add_variable('utra', False, 1, tordtset.data[0][:,:])
pyldtset.write('reco',basedir='.',instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['urec', 'utra'],fmt='vtkh5')
