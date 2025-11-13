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
# CITA PROCEEDINGS MADRID
#
# Last rev: 13/11/2025

# General python imports
import numpy as np
import torch
import cupy  as cp

# General pyLOM inputs
from .utils        import create_dataset
from ..            import Encoder1D, Decoder1D, Encoder1DNoLatent, Decoder1DNoLatent, Autoencoder, VariationalAutoencoder, betaLinearScheduler
from ..            import silu, DEVICE
from ...           import Mesh
from ...vmmath     import temporal_mean, subtract_mean, randomized_qr2, local_randomized_qr, matmul, local_energy
from ...utils      import cr
from ...utils      import mpi_reduce, pprint, MPI_RANK
from ...inp_out    import h5_create_compressed, h5_flush_compressed


## Compute the randomized QR factorization
@cr('GAVI.QR')
def QR(X:np.ndarray,k:int,q:int=1,osampl:int=10):
	r"""
	Function to compute the randomized QR factorization. The wrapped algorithm is the hybrid CPU-GPU from:

	Arnau Miró, Benet Eiximeno, Lucas Gasparino et al. Towards a GPU-enabled billionare SVD in pyLOM, 10 October 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-7678279/v1]

	Args:
		X (np.ndarray): data matrix to factorize
		k (int): number of modes to retain
		q (int, optional): number of power iterations
		osampl (int, optional): number of oversampled modes

	Returns
		[np.ndarray, np.ndarray] the Q and B matrices with k modes each
	
	"""
	r   = k+osampl if k+osampl < X.shape[1] else X.shape[1]
	Xm  = temporal_mean(X)
	X   = subtract_mean(X, Xm)
	Q,B = randomized_qr2(X,r,q)
	
	return Q[:,:k].copy(), B[:k,:].copy()

## Compress the randomized QR factorization
@cr('GAVI.vae_Q')
def vae_Q(fname,Q,mesh,porder,r,nvars,nlayers=1,conv_chan=4,kernel=4,padding=1,func=silu(),epochs=1000,learning_r=5e-3,basedir='./',dtype=np.float32):
	r"""
	Function to compress the Q matrix from the randomized QR factorization following the strategy from CITA PROCEEDINGS MADRID and keeping the same partition as in the running mesh

	Args:
		fname (str): file name where the compressed data will be saved
		Q (np.ndarray): Q matrix to compress
		mesh (Mesh): mesh in which the data is represented
		porder (int): pOrder of the original CFD mesh
		r (int): number of modes to retain from the latent space
		nvars (int): number of variables to compress
		nlayers (int, optional): number of convolutional layers in the autoencoders (default ``1``)
		conv_chan (int, optional): number of convolutional channels in each layer (default ``4``)
		kernel (int, optional): size of the kernel of the convolutions (default ``4``)
		padding (int, optional): size of the padding of the convolutions (default ``1``)
		func (torch.module, optional): activation function (default ``silu()``)
		epochs (int, optional): number of epochs to do the training (default ``1000``)
		learning_r (float, optional): learning rate (default ``5e-3``)
		basedir (str, optional): directory where the compressed file will be saved (default ``./``)
		dtype (np.dtype, optional): data type used to save the arrays (default ``np.float32``)

	"""
	## Get Q dimensions
	nmod    = Q.shape[1]
	## Compute number of AEs to train and points per AE
	nelxAE  = 1*porder**3                             # Compute how many cells we load per autoencoder
	nptxAE  = (porder+1)**3                           # Compute how many points we train in each autoencoder
	nAEs    = int(mesh.ncells/nelxAE)                 # Number of autoencoders in this partition
	nAEsG   = mpi_reduce(nAEs, op='sum', all=True)
	ist,ien = mesh.partition_table.partition_bounds(MPI_RANK, points=False)
	ist,ien = int(ist/nelxAE), int(ien/nelxAE)
	## Define the AE architecture
	activ   = [func for _ in range(nlayers)]
	encoder = Encoder1DNoLatent(nlayers, nmod, nvars, conv_chan, kernel, padding, activ)
	decoder = Decoder1DNoLatent(nlayers, nmod, nvars, conv_chan, kernel, padding, activ)
	vae     = Autoencoder((nmod,), nvars, encoder, decoder, verbose=False)
	## Create the file where the AEs parameters and latents will be saved
	file    = h5_create_compressed(fname, basedir, r, nmod, nvars, nlayers, conv_chan, kernel, nAEsG, nptxAE, dtype)
	Qtrain = np.zeros((nmod,nvars,nptxAE), dtype=np.float32)
	iAE    = 0
	ener_x = 0
	while iAE < nAEs:
		conecE        = mesh.connectivity[iAE*nelxAE:(iAE+1)*nelxAE].flatten()
		_,idx         = np.unique(conecE, return_index=True)
		nodes         = conecE[np.sort(idx)]
		Qtrain[:,0,:] = Q[nodes,:].T
		vae.train()
		datatra, scaler = create_dataset(Qtrain, scale='meanstd')
		vae.fit(datatra, eval_dataset=datatra, batch_size=nptxAE, epochs=epochs, lr=learning_r, BASEDIR='./', pin_memory=False, shuffle=False, conv_loss=1e-2)
		vae.eval()
		latent = vae.latent_space(datatra)
		Q2, B2 = local_randomized_qr(cp.from_dlpack(latent.T), r+10, 1)
		latr   = torch.tensor(matmul(Q2[:,:r],B2[:r,:])).T
		rectrL = vae.decoder(latr)
		ener_x += local_energy(rectrL[:,0,:].T.cpu().detach().numpy()*scaler[1]+scaler[0], Qtrain[:,0,:])
		file   = h5_flush_compressed(file, ist, iAE, scaler, vae, Q2, B2, r)
		iAE += 1
		if np.mod(iAE,1000)==0:
			pprint(0, iAE, ener_x/iAE, flush=True)
	
	file.close()


## Autoencoder on the R
@cr('GAVI.vae_R')
def vae_R(data, latent_dim, nepochs=2500, nlayers=3, conv_chan=64, hid_dim=32, kernel=4, padding=1, func=silu()):
	r"""
	Function to get a disentangled latent representation of the B matrix from the randomized QR factorization:

	Eiximeno, B., A., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025). 
	On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models. 
	Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797

	Args:
		data (np.ndarray): R matrix to compress
		latent_dim (int): number of latent vectors
		nepochs (int, optional): number of epochs to do the training (default ``1000``)
		nlayers (int, optional): number of convolutional layers in the autoencoders (default ``1``)
		conv_chan (int, optional): number of convolutional channels in each layer (default ``4``)
		kernel (int, optional): size of the kernel of the convolutions (default ``4``)
		padding (int, optional): size of the padding of the convolutions (default ``1``)
		func (torch.module, optional): activation function (default ``silu()``)

	Returns:
		Variational autoencoder

	"""

	nmod       = data.shape[2]
	input_chan = data.shape[1]
	activation = [func for _ in range(nlayers + 2)]
	encoder    = Encoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
	decoder    = Decoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
	vae        = VariationalAutoencoder(latent_dim, (nmod,), input_chan, encoder, decoder)
	vae.fit(data, eval_dataset=data, betasch=betaLinearScheduler(0,2.5e-2,500,1000), batch_size=64, epochs=nepochs, lr=5e-4, BASEDIR='./', pin_memory=False)
	return vae