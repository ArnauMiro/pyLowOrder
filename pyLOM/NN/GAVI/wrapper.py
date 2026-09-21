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
import os, json, importlib, datetime, numpy as np
import torch

# General pyLOM inputs
from .utils                            import create_dataset
from ..                                import DEVICE
from ..architectures.encoders_decoders import Encoder1D, Decoder1D, Encoder1DNoLatent, Decoder1DNoLatent
from ..architectures.autoencoders      import Autoencoder, VariationalAutoencoder
from ..utils                           import betaLinearScheduler
from ..utils.activations 			   import silu
from ..dataset						   import Dataset
from ...mesh                           import Mesh
from ...vmmath                         import temporal_mean, subtract_mean, randomized_qr, matmul, local_energy
from ...utils.cr                       import cr, cr_start, cr_stop
from ...utils.parall                   import pprint
from ...utils.mpi                      import mpi_reduce, mpi_barrier, MPI_RANK
from ...utils.gpu                      import cpu_to_gpu, gpu_to_cpu, from_dlpack
from ...utils.errors                   import raiseError, raiseWarning
from ...inp_out.io_h5                  import h5_create_compressed, h5_flush_compressed


## GAVI R-VAE configuration I/O
GAVI_R_CONFIG_VERSION = 1


def _activation_to_str(func) -> str:
	r'''Serialise an activation instance as an import path, e.g. "torch.nn.SiLU".'''
	cls = type(func)
	if getattr(torch.nn, cls.__name__, None) is cls:
		return 'torch.nn.%s' % cls.__name__
	return '%s.%s' % (cls.__module__, cls.__name__)


def _activation_from_str(path:str):
	r'''Rebuild an activation from its import path. Default-constructed only.'''
	module, _, name = path.rpartition('.')
	return getattr(importlib.import_module(module), name)()


def _gavi_paths(BASEDIR:str, modelstr:str, latent_dim:int):
	r'''Checkpoint and configuration paths, sharing the stem written by ``fit``.'''
	stem = os.path.join(BASEDIR.rstrip('/'), '%s_%i' % (modelstr, latent_dim))
	return stem + '.pth', stem + '.json'


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
def vae_R(data:Dataset, latent_dim:int, *, eval_data:Dataset=None, nepochs:int=2500, nlayers:int=3, conv_chan:int=64, hid_dim:int=32, kernel:int=4, padding:int=1, func:object=silu(), 
		batch_size:int=64, lr:float=5e-4, beta_start:float=0.0, beta_end:float=2.5e-2, beta_first:int=500, beta_last:int=1000, seed:int=None, BASEDIR:str='./', modelstr='gavi_R_latent'):
	r"""
	Function to get a disentangled latent representation of the B matrix from the
	randomized QR factorization.

	Eiximeno, B., Miro, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025).
	On the integration of geometry agnostic variational-autoencoders into large-scale
	SVD based models. Computers & Fluids, 302, 106797.

	Args:
		data (Dataset): R matrix to compress.
		latent_dim (int): number of latent vectors.
		eval_data (Dataset, optional): held-out dataset for the validation loss.
			``None`` (default) disables validation. Passing ``data`` here reports
			the training loss twice and cannot detect overfitting.
		nepochs (int, optional): number of training epochs (default ``2500``).
		beta_start, beta_end, beta_first, beta_last: linear KL schedule. ``beta``
			is ``beta_start`` until epoch ``beta_first`` and reaches ``beta_end``
			at epoch ``beta_last``; a run shorter than ``beta_last`` never
			reaches ``beta_end``.
		seed (int, optional): seed for weight initialisation and the
			reparameterisation draw (default ``None``, unseeded).
		...

	Returns:
		VariationalAutoencoder
	"""
	if seed is not None:
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	nmod       = data.shape[2]
	input_chan = data.shape[1]
	activation = [func for _ in range(nlayers + 2)]
	encoder    = Encoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
	decoder    = Decoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
	vae        = VariationalAutoencoder(latent_dim, (nmod,), input_chan, encoder, decoder)
	vae.gavi_config = dict(
		latent_dim = latent_dim,
		nmod       = nmod,
		input_chan = input_chan,
		nlayers    = nlayers,
		conv_chan  = conv_chan,
		hid_dim    = hid_dim,
		kernel     = kernel,
		padding    = padding,
		batch_norm = False,
		activation = _activation_to_str(func),
		training   = dict(nepochs=nepochs, lr=lr, batch_size=batch_size,
		                  beta_start=beta_start, beta_end=beta_end,
		                  beta_first=beta_first, beta_last=beta_last,
		                  seed=seed, validated=eval_data is not None),
	)

	betasch    = betaLinearScheduler(beta_start, beta_end, beta_first, beta_last)
	vae.fit(data, eval_dataset=eval_data, betasch=betasch, batch_size=batch_size, epochs=nepochs, lr=lr, BASEDIR=BASEDIR, pin_memory=False, MODELSTR="%s_%i" % (modelstr, latent_dim))
	save_vae_R(vae, BASEDIR=BASEDIR, modelstr=modelstr, save_weights=False)
	return vae


@cr('GAVI.save_vae_R')
def save_vae_R(vae, BASEDIR:str='./', modelstr:str='gavi_R_latent', save_weights:bool=False, **extra):
	r"""
	Save a GAVI R-VAE configuration so it can be reloaded without restating the
	architecture.

	Args:
		vae (VariationalAutoencoder): model returned by :func:`vae_R`, which carries
			its own description in ``vae.gavi_config``.
		BASEDIR (str, optional): output folder (default ``"./"``).
		modelstr (str, optional): file stem (default ``"gavi_R_latent"``).
		save_weights (bool, optional): also write the ``.pth``. ``fit`` already
			writes it under the same stem, so this is only needed after
			``fine_tune`` or manual surgery (default ``False``).
		**extra: extra JSON-serialisable provenance stored under ``"extra"``
			(fold index, split definition, scalar scalers, ...). Arrays do not
			belong here; keep those in the companion ``.npz``.

	Returns:
		str: path to the JSON file written.
	"""
	if not hasattr(vae, 'gavi_config'):
		raiseError('This model carries no gavi_config: it was not built by GAVI.vae_R, or predates the configuration-saving change.')
	cfg              = dict(vae.gavi_config)
	pthfile, cfgfile = _gavi_paths(BASEDIR, modelstr, cfg['latent_dim'])
	cfg['config_version'] = GAVI_R_CONFIG_VERSION
	cfg['checkpoint']     = os.path.basename(pthfile)
	cfg['torch_version']  = torch.__version__
	cfg['saved']          = datetime.datetime.now().isoformat(timespec='seconds')
	if extra: cfg['extra'] = {**cfg.get('extra', {}), **extra}

	os.makedirs(os.path.dirname(cfgfile) or '.', exist_ok=True)
	if save_weights: torch.save(vae.state_dict(), pthfile)
	with open(cfgfile, 'w') as f:
		json.dump(cfg, f, indent=2, sort_keys=True)
	pprint(0, 'GAVI R-VAE configuration written to %s' % cfgfile, flush=True)
	return cfgfile

@cr('GAVI.load_vae_R')
def load_vae_R(latent_dim:int, *, nlayers:int=None, conv_chan:int=None, hid_dim:int=None, kernel:int=None, padding:int=None, func:object=None,
               BASEDIR:str='./', modelstr:str='gavi_R_latent', strict:bool=True):
	r"""
	Load a trained GAVI R-VAE from a saved state_dict (no training).

	When ``<BASEDIR>/<modelstr>_<latent_dim>.json`` exists the architecture is read
	from it and nothing else is required::

		vae = pyLOM.NN.GAVI.load_vae_R(BASEDIR='gavi_folds/fold00')

	If ``latent_dim`` is omitted the folder is globbed and must contain exactly one
	configuration. When no configuration is found the previous behaviour applies:
	``data`` and ``latent_dim`` become mandatory and the architecture defaults are
	used, with a warning, since nothing then guarantees the network matches the
	weights.

	Args:
		latent_dim (int, optional): number of latent vectors.
		nlayers, conv_chan, hid_dim, kernel, padding, func (optional): override the
			stored architecture. A warning is raised on disagreement.
		BASEDIR (str, optional): folder holding the ``.pth`` and ``.json``.
		modelstr (str, optional): file stem (default ``"gavi_R_latent"``).
		strict (bool, optional): passed to ``load_state_dict`` (default ``True``).

	Returns:
		VariationalAutoencoder: model with the weights loaded, in eval mode, with
		``vae.gavi_config`` reattached.
	"""
	# Locate the configuration
	_, cfgfile = _gavi_paths(BASEDIR, modelstr, latent_dim)
	with open(cfgfile, 'r') as f:
			cfg = json.load(f)
	given = dict(nlayers=nlayers, conv_chan=conv_chan, hid_dim=hid_dim, kernel=kernel, padding=padding)
	keys = ('latent_dim','nmod','input_chan','nlayers','conv_chan','hid_dim','kernel','padding')
	arch = {k: cfg[k] for k in keys}
	arch['batch_norm'] = cfg.get('batch_norm', False)
	arch['activation'] = cfg['activation']
	for k, v in given.items():
		if v is not None and v != arch[k]:
			raiseWarning('%s=%r overrides the stored value %r' % (k, v, arch[k]))
			arch[k] = v
	if func is not None: arch['activation'] = _activation_to_str(func)
	# Rebuild and load
	activ      = _activation_from_str(arch['activation'])
	activation = [activ for _ in range(arch['nlayers'] + 2)]
	encoder    = Encoder1D(arch['nlayers'], arch['latent_dim'], arch['nmod'], arch['input_chan'],
	                       arch['conv_chan'], arch['kernel'], arch['padding'], activation,
	                       arch['hid_dim'], batch_norm=arch['batch_norm'])
	decoder    = Decoder1D(arch['nlayers'], arch['latent_dim'], arch['nmod'], arch['input_chan'],
	                       arch['conv_chan'], arch['kernel'], arch['padding'], activation,
	                       arch['hid_dim'], batch_norm=arch['batch_norm'])
	vae        = VariationalAutoencoder(arch['latent_dim'], (arch['nmod'],), arch['input_chan'],
	                                    encoder, decoder)
	ckpt, _    = _gavi_paths(BASEDIR, modelstr, arch['latent_dim'])
	vae.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=strict)
	vae.eval()
	vae.gavi_config = cfg if cfg is not None else arch
	return vae