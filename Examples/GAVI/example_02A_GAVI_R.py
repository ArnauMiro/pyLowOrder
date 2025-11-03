import torch
import pyLOM, pyLOM.NN

#import gavi
import numpy as np

device = pyLOM.NN.select_device() #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
FILE       = 'QR.h5'
COMPILED   = False
MIXED      = True
nlayers    = 2
input_chan = 1
conv_chan  = 16
kernel     = 4
padding    = 1
activation = [pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu()]
latent_dim = 3
hid_dim    = 32

# Load data
R     = pyLOM.GAVI.load(FILE, vars=['B'])[0][:1600,:]
nmod  = R.shape[0]
nt    = R.shape[1]
Rmax  = np.abs(np.max(R))
R     = torch.tensor((R/ Rmax).astype(np.float32), device=device)

data    = pyLOM.NN.Dataset(R)
encoder = pyLOM.NN.Encoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
decoder = pyLOM.NN.Decoder1D(nlayers, latent_dim, nmod, input_chan, conv_chan, kernel, padding, activation, hid_dim, batch_norm=False)
vae     = pyLOM.NN.VariationalAutoencoder(latent_dim, (nmod,), input_chan, encoder, decoder, device)
vae.fit(data, eval_dataset=data, batch_size=int(nt/4), epochs=500, lr=1e-3, BASEDIR='./', pin_memory=False)

rectra = vae.reconstruct(data)
corr, detR = vae.correlation(data)

print(detR)

'''

if COMPILED:
	encoder = torch.compile(encoder, mode="max-autotune")
	decoder = torch.compile(decoder, mode="max-autotune")
	vae     = torch.compile(gavi.VariationalAutoencoder(latent_dim,(nh,nw), input_chan, encoder, decoder, device, compiled=COMPILED), mode="max-autotune")
	compstr = 'compiled'
else:
	vae     = gavi.VariationalAutoencoder(latent_dim, (nh,nw), input_chan, encoder, decoder, device)
	compstr = 'noncompiled'

if MIXED:
	mixstr = 'fpmix'
else:
	mixstr = 'fp32'



latent = vae.latent_space(datatra)
if latent_dim > 1:  


np.save('latents/CPUlatentnewVanilla_%s_%s_%i_%i_%.2e' % (compstr,mixstr,latent_dim,hid_dim,beta), latent.cpu().numpy())
np.save('reconstructions_R/CPURnewVanilla_%s_%s_%i_%i_%.2e' % (compstr,mixstr,latent_dim,hid_dim,beta), rectra.cpu().numpy())

print('Recovered energy:', energy*100)
'''

pyLOM.cr_info()