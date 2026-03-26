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
import os, numpy as np
import torch

# General pyLOM inputs
from .utils                            import create_dataset
from ..                                import DEVICE
from ..architectures.encoders_decoders import Encoder1D, Decoder1D, Encoder1DNoLatent, Decoder1DNoLatent
from ..architectures.autoencoders      import Autoencoder, VariationalAutoencoder
from ..utils                           import silu, Dataset, betaLinearScheduler
from ...mesh                           import Mesh
from ...vmmath                         import temporal_mean, subtract_mean, randomized_qr, matmul, local_energy
from ...utils.cr                       import cr, cr_start, cr_stop
from ...utils.parall                   import pprint
from ...utils.mpi                      import mpi_reduce, mpi_barrier, MPI_RANK
from ...utils.gpu                      import cpu_to_gpu, gpu_to_cpu, from_dlpack
from ...inp_out.io_h5                  import h5_create_compressed, h5_flush_compressed


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
	Q,B = randomized_qr(X,r,q,hybrid=True)
	
	return Q[:,:k].copy(), B[:k,:].copy()

## Compress the randomized QR factorization
@cr('GAVI.vae_Q')
def vae_Q(fname:str,Q:tuple,mesh:Mesh,porder:int,r:int,nlayers:int=1,conv_chan:int=4,kernel:int=4,padding:int=1,func:object=silu(),epochs:int=1000,learning_r:float=5e-3,basedir:str='./',dtype:np.dtype=np.float32):
	r"""
	Function to compress the Q matrix from the randomized QR factorization following the strategy from CITA PROCEEDINGS MADRID and keeping the same partition as in the running mesh

	Args:
		fname (str): file name where the compressed data will be saved
		Q (tuple): Q matrices to compress
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
	nmod    = Q[0].shape[1]
	nvars   = len(Q)
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
	h5_create_compressed(fname, basedir, r, nmod, nvars, nlayers, conv_chan, kernel, nAEsG, nptxAE, dtype)
	means   = np.zeros((nAEs,nvars), dtype=dtype)
	stds    = np.zeros((nAEs,nvars), dtype=dtype)
	weights = torch.zeros((nAEs,conv_chan,nvars,kernel), device=DEVICE)
	biases  = torch.zeros((nAEs,nvars), device=DEVICE)
	Qs      = cpu_to_gpu(np.zeros((nAEs,int(nmod/2**nlayers)*conv_chan,r), dtype=dtype))
	Bs      = cpu_to_gpu(np.zeros((nAEs,r,nptxAE), dtype=dtype))
	Qtrain  = np.zeros((nmod,nvars,nptxAE), dtype=np.float32)
	ener    = np.zeros((nvars,), dtype=np.float32)
	for iAE in range(nAEs):
		conecE        = mesh.connectivity[iAE*nelxAE:(iAE+1)*nelxAE].flatten()
		_,idx         = np.unique(conecE, return_index=True)
		nodes         = conecE[np.sort(idx)]
		for ivar in range(nvars):
			Qtrain[:,ivar,:] = Q[ivar][nodes,:].T
		vae.train()
		datatra, scaler = create_dataset(Qtrain, scale='meanstd')
		vae.fit(datatra, eval_dataset=None, batch_size=nptxAE, epochs=epochs, lr=learning_r, BASEDIR='./', pin_memory=False, shuffle=False, conv_loss=1e-2)
		vae.eval()
		latent  = vae.latent_space(datatra)
		Q2, B2  = randomized_qr(from_dlpack(latent.T), r+10, 1,local=True)
		latr    = torch.tensor(matmul(Q2[:,:r],B2[:r,:])).T
		rectrL  = vae.decoder(latr)
		for ivar in range(nvars):
			ener[ivar] += local_energy(rectrL[:,ivar,:].T.cpu().detach().numpy()*scaler[ivar,1]+scaler[ivar,0], Qtrain[:,ivar,:])
		if np.mod(iAE,1000)==0:
			pprint(0, iAE, ener/iAE, flush=True)
		means[iAE] = scaler[:,0]
		stds[iAE]  = scaler[:,1]
		weights[iAE,:,:,:] = vae.state_dict()['decoder.deconv_layers.0.weight'].detach().clone()
		biases[iAE,:]      = vae.state_dict()['decoder.deconv_layers.0.bias'].detach().clone()
		Qs[iAE] = Q2[:,:r]
		Bs[iAE] = B2[:r,:]
	
	mpi_barrier()
	h5_flush_compressed(fname, basedir, ist, ien, means, stds, weights.detach().cpu().numpy(), biases.detach().cpu().numpy(), gpu_to_cpu(Qs.get()), gpu_to_cpu(Bs.get()))
	

## Reconstruct_Q
@cr('GAVI.reconstruct_Q')
def reconstruct_Q(mesh:Mesh,nelxAE:int,nmod:int,Qmeans:np.ndarray,Qstds:np.ndarray,weights:torch.tensor,biases:torch.tensor,Qs:np.ndarray,Bs:np.ndarray,ivar:int=0,padding:int=1,func:object=silu()):
	r"""
	Function to reconstruct the compressed data of Q
	
	Args:
		mesh (Mesh): mesh in which we will represent the reconstructed data
		nelxAE (int): number of elements learnt by each autoencoder
		nmod (int): number of modes
		Qmeans (np.ndarray): mean Q value of the input data of each autoencoder. Has as many columns as compressed variables
		Qstds (np.ndarray): standard deviation of Q value of the input data of each autoencoder. Has as many columns as compressed variables.
		weights (torch.tensor): weights of the decoder
		biases (torch.tensor): biases of the decoder
		Qs (np.ndarray): orthogonal matrix of the factorized latent vectors at each autoencoder
		Bs (np.ndarray): reduced matrix of the factorized latent vectors at each autoencoder
		ivar (int, optional): index of the decompressed variable, the output channel that we'll get (default ``0``)
		padding (int, optional): amount of padding in the convolutions (default ``1``)
		func (object, optional): activation function of the decoder layers (default ``silu()``)
		
	Returns:
		np.ndarray: reconstructed Q of the variable stored in the ivar channel.	
	"""
	nAEs      = Qmeans.shape[0]
	nvars     = Qmeans.shape[1]
	conv_chan = weights.shape[1]
	kernel    = weights.shape[3]
	nlayers   = int(np.log2(Qs.shape[1]/nmod))
	activ     = [func for _ in range(nlayers)]
	decoder = Decoder1DNoLatent(nlayers, nmod, nvars, conv_chan, kernel, padding, activ)
	decoder.to(DEVICE)
	Q = np.zeros((mesh.xyz.shape[0],nmod))
	for iel in range(nAEs):
		# Get global node numbering
		conecE = mesh.connectivity[iel*nelxAE:(iel+1)*nelxAE]
		_,idx  = np.unique(conecE.flatten(), return_index=True)
		nodes  = conecE.flatten()[np.sort(idx)]
		lat = matmul(Qs[iel,:,:], Bs[iel,:,:])
		lat = torch.tensor(lat).T
		cr_start('GAVI.decode', 0)
		with torch.no_grad():
			decoder.deconv_layers[0].weight.copy_(weights[iel])
			decoder.deconv_layers[0].bias.copy_(biases[iel])
			out = decoder(lat)
		cr_stop('GAVI.decode', 0)
		Q[nodes] = (out[:,ivar,:].detach().cpu().numpy()*Qstds[iel,ivar]+Qmeans[iel,ivar])
	
	return Q

## Autoencoder on the R
@cr('GAVI.vae_R')
def vae_R(data:Dataset, latent_dim:int, *, nepochs:int=2500, nlayers:int=3, conv_chan:int=64, hid_dim:int=32, kernel:int=4, padding:int=1, func:object=silu(), BASEDIR:str='./', modelstr='gavi_R_latent'):
	r"""
	Function to get a disentangled latent representation of the B matrix from the randomized QR factorization:

	Eiximeno, B., A., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025). 
	On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models. 
	Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797

	Args:
		data (Dataset): R matrix to compress
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
	vae.fit(data, eval_dataset=data, betasch=betaLinearScheduler(0,2.5e-2,500,1000), batch_size=64, epochs=nepochs, lr=5e-4, BASEDIR=BASEDIR, pin_memory=False, MODELSTR="%s_%i" % (modelstr,latent_dim))
	return vae


@cr('GAVI.load_vae_R')
def load_vae_R(data:Dataset, latent_dim:int, *, nlayers:int=3, conv_chan:int=64, hid_dim:int=32, kernel:int=4, padding:int=1, func:object=silu(), BASEDIR:str='./', modelstr='gavi_R_latent'):
	r"""
	Load a trained GAVI R-VAE from a saved state_dict (same architecture as vae_R, no training).

	Eiximeno, B., A., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025).
	On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models.
	Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797

	Args:
		data (Dataset): Dataset with same shape as used for training (used only for nmod, input_chan).
		latent_dim (int): number of latent vectors (must match the saved model).
		nlayers (int, optional): number of convolutional layers (default ``3``).
		conv_chan (int, optional): number of convolutional channels (default ``64``).
		hid_dim (int, optional): hidden dimension (default ``32``).
		kernel (int, optional): kernel size (default ``4``).
		padding (int, optional): padding (default ``1``).
		func (object, optional): activation function (default ``silu()``).
		BASEDIR (str, optional): directory containing ``gavi_R_latent_<latent_dim>.pth`` (default ``"./"``).

	Returns:
		VariationalAutoencoder: VAE with loaded state_dict, in eval mode.
	"""
	nmod       = data.shape[2]
	input_chan = data.shape[1]
	activation = [func for _ in range(nlayers + 2)]
	encoder    = Encoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
	decoder    = Decoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
	vae        = VariationalAutoencoder(latent_dim, (nmod,), input_chan, encoder, decoder)
	ckpt_path  = os.path.join(BASEDIR.rstrip('/'),'%s_%i.pth' % (modelstr,latent_dim))
	vae.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
	vae.eval()
	return vae