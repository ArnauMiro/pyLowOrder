import numpy as np
import torch
import pyLOM, pyLOM.NN
import h5py
import cupy as cp

from   pyLOM.utils import MPI_COMM, MPI_RANK

## Architecture Parameters
FILE       = 'QR.h5'
COMPILED   = False
MIXED      = False
RESTART    = False
nlayers    = 1
input_chan = 4
conv_chan  = 4
kernel     = 4
padding    = 1
activation = [pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu()]
epochs     = 1000
learning_r = 5e-3
mixstr     = 'fpmix' if MIXED else 'fp32'


## Load data
vlist   = ['velox']
mesh    = pyLOM.Mesh.load('../pylom_mesh.h5')
porder  = 4                                       ## Input the pOrder of the original mesh to group the rest of elements
nelxAE  = 1*porder**3                             ## Compute how many cells we load per autoencoder
nptxAE  = (porder+1)**3                           ## Compute how many points we train in each autoencoder
nAEs    = int(mesh.ncells/nelxAE)                 ## Number of autoencoders in this partition
nAEsG   = pyLOM.utils.mpi_reduce(nAEs, op='sum', all=True)
ist,ien = mesh.partition_table.partition_bounds(MPI_RANK, points=False)
ist,ien = int(ist/nelxAE), int(ien/nelxAE)
nvars   = len(vlist)
r       = 10

Q      = pyLOM.NN.GAVI.load(FILE, vars=['Q'], ptable=mesh.partition_table)[0]
nmod   = Q.shape[1]

encoder = pyLOM.NN.Encoder1DNoLatent(nlayers, nmod, nvars, conv_chan, kernel, padding, activation)
decoder = pyLOM.NN.Decoder1DNoLatent(nlayers, nmod, nvars, conv_chan, kernel, padding, activation)
vae     = pyLOM.NN.Autoencoder((nmod,), nvars, encoder, decoder, verbose=False)

pyLOM.cr_start('create_groups', 0)
file  = h5py.File('compressed_data_array.h5', mode="w", driver='mpio', comm=MPI_COMM)
stats = file.create_group("STATS")
stats.create_dataset("mean", shape=(nAEsG,nvars), dtype=np.float32)
stats.create_dataset("std",  shape=(nAEsG,nvars), dtype=np.float32)
decod = file.create_group("DECODER")
decod.create_dataset("weights", shape=(nAEsG,conv_chan,nvars,kernel), dtype=np.float32)
decod.create_dataset("biases", shape=(nAEsG,nvars), dtype=np.float32)
lats  = file.create_group("LATENTS")
lats.create_dataset("U",  shape=(nAEsG,int(nmod/2)*conv_chan,r), dtype=np.float32)
lats.create_dataset("SV", shape=(nAEsG,r,nptxAE), dtype=np.float32)

pyLOM.cr_stop('create_groups', 0)
pyLOM.pprint(0, 'Groups created', flush=True)

Qtrain = np.zeros((nmod,nvars,nptxAE), dtype=np.float32)
iAE    = 0
while iAE < nAEs:
	conecE        = mesh.connectivity[iAE*nelxAE:(iAE+1)*nelxAE].flatten()
	_,idx         = np.unique(conecE, return_index=True)
	nodes         = conecE[np.sort(idx)]
	Qtrain[:,0,:] = Q[nodes,:].T
	vae.train()
	datatra, scaler = pyLOM.NN.GAVI.create_dataset(Qtrain, scale='meanstd')
	vae.fit(datatra, eval_dataset=datatra, batch_size=nptxAE, epochs=epochs, lr=learning_r, BASEDIR='./', pin_memory=False, shuffle=False)
	vae.eval()
	latent = vae.latent_space(datatra)
	Q2, B2 = pyLOM.math.randomized_qr(cp.from_dlpack(latent.T), r+10, 1)
	latr   = torch.tensor(pyLOM.math.matmul(Q2[:,:r],B2[:r,:])).T
	rectrL = vae.decoder(latr)
	ener_x = pyLOM.math.energy(rectrL[:,0,:].T.cpu().detach().numpy()*scaler[1]+scaler[0], Qtrain[:,0,:])
	if ener_x > 0.8:
		pyLOM.pprint(-1, iAE, ener_x, flush=True)
		pyLOM.cr_start('flush_hdf5', 0)
		file['STATS/mean'][ist+iAE,:] = scaler[0]
		file['STATS/std'][ist+iAE,:]  = scaler[1]
		file['DECODER/weights'][ist+iAE,:,:,:] = vae.state_dict()['decoder.deconv_layers.0.weight'].detach().cpu().numpy()
		file['DECODER/biases'][ist+iAE,:]      = vae.state_dict()['decoder.deconv_layers.0.bias'].detach().cpu().numpy()
		file['LATENTS/U'][ist+iAE,:,:]  = Q2[:,:r].get()
		file['LATENTS/SV'][ist+iAE,:,:] = B2[:r,:].get()
		pyLOM.cr_stop('flush_hdf5', 0)
		iAE += 1
	else:
		iAE = iAE  

file.close()

pyLOM.cr_info()