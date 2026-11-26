#!/usr/bin/env python
#
# PYLOM Testsuite
# Fast checks of the NN module standards: dataset shapes and precision,
# scalers, schedulers, model precision, VAE modes, cr blocks and GAVI utilities.
# Every check runs on small synthetic data, so the whole case takes a few seconds.
#
# Last revision: 25/11/2026
import sys, os, numpy as np, torch
import pyLOM, pyLOM.NN
from pyLOM.utils.cr        import _findch
from pyLOM.NN.GAVI.utils   import _same_precision

DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
SEED      = 42

device = pyLOM.NN.select_device('cpu')
pyLOM.NN.set_seed(SEED)
rng    = np.random.default_rng(SEED)

def bits(dtype):
	'''Size in bits of a numpy or torch dtype: 32 or 64'''
	return torch.empty(0, dtype=dtype).element_size()*8 if isinstance(dtype, torch.dtype) else np.dtype(dtype).itemsize*8


## Synthetic data: N snapshots of a nh x nw field, stored by columns
nh, nw, N = 8, 6, 5
x64 = rng.random((nh*nw, N))
x32 = x64.astype(np.float32)


## Dataset: the precision of the input is kept
pyLOM.pprint(0,'TSUITE dataset_bits    =',bits(pyLOM.NN.Dataset((x64,),(nh,nw))[:].dtype),bits(pyLOM.NN.Dataset((x32,),(nh,nw))[:].dtype))


## Dataset: the singleton channel axis is squeezed by default and kept with squeeze_last_dim=False
s_def  = pyLOM.NN.Dataset((x32,),(nh,nw))[:].shape
s_keep = pyLOM.NN.Dataset((x32,),(nh,nw),squeeze_last_dim=False)[:].shape
s_two  = pyLOM.NN.Dataset((x32,x32),(nh,nw))[:].shape
pyLOM.pprint(0,'TSUITE dataset_squeeze =',len(s_def),*s_def)
pyLOM.pprint(0,'TSUITE dataset_keep    =',len(s_keep),*s_keep)
pyLOM.pprint(0,'TSUITE dataset_2chan   =',len(s_two),*s_two)


## Scalers: fit returns the scaler itself
a = rng.random((20,3))
fit_self = [int(s.fit(a) is s) for s in (pyLOM.NN.MinMaxScaler(), pyLOM.NN.StandardScaler(), pyLOM.NN.RobustScaler())]
pyLOM.pprint(0,'TSUITE scaler_fit_self =',*fit_self)


## StandardScaler: population std (divide by n) for numpy arrays and tensors alike
sn = pyLOM.NN.StandardScaler().fit(a).variable_scaling_params[0]['std']
st = pyLOM.NN.StandardScaler().fit(torch.from_numpy(a)).variable_scaling_params[0]['std']
pyLOM.pprint(0,'TSUITE standard_std    =',sn,st,np.std(a[:,0]))


## MinMaxScaler: scaling per column (default) or per row (column=True)
b    = np.array([[0.,10.,100.],[1.,20.,300.]])
rcol = pyLOM.NN.MinMaxScaler().fit_transform(b)
rrow = pyLOM.NN.MinMaxScaler(column=True).fit_transform(b)
pyLOM.pprint(0,'TSUITE minmax_default  =',*np.asarray(rcol).ravel())
pyLOM.pprint(0,'TSUITE minmax_column   =',*np.asarray(rrow).ravel())


## Beta scheduler: start_value before start_epoch, linear up to warmup, end_value after
sch = pyLOM.NN.betaLinearScheduler(start_value=0.1, end_value=1.0, start_epoch=5, warmup=15)
pyLOM.pprint(0,'TSUITE beta_scheduler  =',sch.getBeta(0),sch.getBeta(5),sch.getBeta(10),sch.getBeta(20))


## Model precision: a float32 model fed with a float64 dataset predicts in float32
X  = rng.random((64,2))
y  = np.sin(X[:,0]) + X[:,1]**2
td = pyLOM.NN.Dataset(variables_out=(y,), variables_in=X)
mlp = pyLOM.NN.MLP(input_size=2, output_size=1, n_layers=2, hidden_size=8, device=device, seed=SEED, verbose=False)
p_mlp = mlp.predict(td)
rbf = pyLOM.NN.RBFELM(input_size=2, output_size=1, n_centers=10, device=device, seed=SEED, verbose=False)
rbf.fit(td)
p_rbf = rbf.predict(td)
pyLOM.pprint(0,'TSUITE model_bits      =',bits(td[:][0].dtype),bits(p_mlp.dtype),bits(p_rbf.dtype))


## VAE modes: shape N*(latent_dim+1), mode 1 is D(0) and mode i+1 is D(e_i)-D(0)
lat_dim, nl, ch, vh, vw = 3, 2, 4, 8, 8 # the grid must be divisible by 2**nlayers
enc  = pyLOM.NN.Encoder2D(nl, lat_dim, vh, vw, 1, ch, 4, 1, [torch.nn.ELU()]*(nl+2), 8, vae=True)
dec  = pyLOM.NN.Decoder2D(nl, lat_dim, vh, vw, 1, ch, 4, 1, [torch.nn.ELU()]*(nl+2), 8)
vae  = pyLOM.NN.VariationalAutoencoder(lat_dim, (vh,vw), 1, enc, dec, device=device)
md   = vae.modes(ivar=1).reshape((vh*vw, lat_dim+1))
d0   = vae.decode(np.zeros((1,lat_dim)))[0][:,0]
d1   = vae.decode(np.eye(lat_dim)[:1])[0][:,0]
pyLOM.pprint(0,'TSUITE vae_modes       =',vae.modes(ivar=1).size,np.abs(md[:,0]-d0).max(),np.abs(md[:,1]-(d1-d0)).max())


## cr blocks: the channel stops even when the block raises
for name, block in (('tsuite_cr_block', pyLOM.cr_block), ('tsuite_cr_nvtx_block', pyLOM.cr_nvtx_block)):
	try:
		with block(name):
			raise ValueError('expected')
	except ValueError:
		pass
cr_running = [int(_findch(n).is_running()) for n in ('tsuite_cr_block','tsuite_cr_nvtx_block')]
pyLOM.pprint(0,'TSUITE cr_running      =',*cr_running)


## GAVI: one dtype for the compiled routines, float32 if any input is float32
pyLOM.pprint(0,'TSUITE gavi_precision  =',*[bits(o.dtype) for o in _same_precision(x32, x64)],*[bits(o.dtype) for o in _same_precision(x64, x64)])


## GAVI: meanstd scaling keeps the data precision and uses the population std
q = rng.random((10,2,N))
ds, sc = pyLOM.NN.GAVI.create_dataset((q[:,0,:],q[:,1,:]), scale='meanstd', device=device)
pyLOM.pprint(0,'TSUITE gavi_meanstd    =',bits(ds[:].dtype),float(sc[0,1]),np.std(q[:,0,:]))


## GAVI: energy accepts a float64 dataset with a float32 reconstruction (the compiled build
## rejects mixed precision); a reconstruction at 90% of the data recovers 1-0.1**2 = 0.99
dq, _ = pyLOM.NN.GAVI.create_dataset((q[:,0,:],), scale=None, device=device)
rec   = (0.9*q[:,0,:])[np.newaxis].astype(np.float32) # (channels, modes, samples)
pyLOM.pprint(0,'TSUITE gavi_energy     =',float(np.max(np.abs(pyLOM.NN.GAVI.energy(dq, rec, 0)))))


pyLOM.cr_info()
pyLOM.pprint(0,'End of output')
