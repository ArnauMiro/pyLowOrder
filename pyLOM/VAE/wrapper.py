import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np

from   torch.utils.tensorboard import SummaryWriter
from   torchsummary            import summary

## Wrapper of the activation functions
def tanh():
    return nn.Tanh()

def relu():
    return nn.ReLU()

def elu():
    return nn.ELU()

def sigmoid():
    return nn.Sigmoid()

def leakyRelu():
    return nn.LeakyReLU()

## Wrapper of the Dataset class

## Wrapper of a variational autoencoder
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, nx, ny, encoder, decoder, device='cpu'):
        super(VariationalAutoencoder, self).__init__()
        self.lat_dim = np.int(latent_dim)
        self.nx      = nx
        self.ny      = ny
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        encoder.to(self._device)
        decoder.to(self._device)
        self.to(self._device)
        summary(self, input_size=(1, self.nx, self.ny))
   
    def _reparamatrizate(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.rand_like(std)  #we create a normal distribution (0 ,1 ) with the dimensions of std        
        sample = mu + std*epsilon
        return  sample
    
    def _lossfunc(self, x, recon_x):
        return F.mse_loss(recon_x.view(-1, self.nx*self.ny), x.view(-1, self.nx*self.ny),reduction='mean')
        
    def _kld(self, mu, logvar):    
        return 0.5*torch.mean(1 + logvar - mu**2 - logvar.exp())
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._reparamatrizate(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
       
    def train_model(self, train_data, vali_data, beta, nepochs, callback=None, learning_rate=3e-4, BASEDIR='./'):
        prev_train_loss = 1e99
        writer = SummaryWriter(BASEDIR)
        for epoch in range(nepochs):
            self.train()
            num_batches = 0 
            learning_rate = learning_rate * 1/(1 + 0.001 * epoch)               #HARDCODED!!
            optimizer = torch.optim.Adam(self.parameters(), lr= learning_rate)  #HARDCODED!!
            tr_loss = 0
            mse     = 0
            kld     = 0
            for batch in train_data:
                print(batch.shape)
                batch = batch.to(self._device)
                recon, mu, logvar, _ = self(batch)
                mse_i  = self._lossfunc(batch, recon)
                bkld_i = self._kld(mu,logvar)*beta
                loss   = mse_i - bkld_i
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                mse     += self._lossfunc(batch, recon).item()
                kld     += self._kld(mu,logvar).item()*beta
                num_batches += 1
            with torch.no_grad():
                val_batches = 0
                va_loss     = 0
                for val_batch in vali_data:
                    val_batch = val_batch.to(self._device)
                    val_recon, val_mu, val_logvar , _ = self(val_batch)
                    mse_i       = self._lossfunc(val_batch, val_recon)
                    bkld_i      = self._kld(val_mu,val_logvar)*beta
                    vali_loss   = mse_i - bkld_i
                    va_loss     += vali_loss.item()
                    val_batches += 1
                tr_loss/=num_batches
                va_loss/=val_batches
                mse /= num_batches
                kld /= num_batches
                writer.add_scalar("Loss/train",tr_loss,epoch+1)
                writer.add_scalar("Loss/vali", va_loss,epoch+1)
                writer.add_scalar("Loss/mse",  mse,    epoch+1)
                writer.add_scalar("Loss/kld",  kld,    epoch+1)

            if callback.early_stop(va_loss, prev_train_loss, tr_loss):
                print('Early Stopper Activated at epoch %i' %epoch, flush=True)
                break
            prev_train_loss = tr_loss   
            print('Epoch [%d / %d] average training error: %.5e' % (epoch+1, nepochs, tr_loss), flush=True)
        writer.flush()
        writer.close()
        torch.save(self.state_dict(), '%s/model_state' % BASEDIR)

            
    def reconstruct(self, dataset):
        ##  Compute reconstruction and its accuracy
        ek     = np.zeros((len(dataset),))
        rec    = np.zeros((self.nx*self.ny,len(dataset))) 
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            ## Energy recovered in reconstruction
            instant = iter(loader)
            energy_batch = next(instant)
            for i in range(len(dataset)):
                x = energy_batch[i,0,:,:]
                x = torch.reshape(x, [1,1,self.nx,self.ny])
                x = x.to(self._device)
                x_recon  = self(x)
                x_recon  = np.asanyarray(x_recon[0].cpu())
                x_recon  = x_recon[0,0,:,:]
                x_recon  = torch.reshape(torch.tensor(x_recon),[self.nx*self.ny, 1])
                rec[:,i] = x_recon.detach().numpy()[:,0]
                x = torch.reshape(x,[self.nx*self.ny,1])
                x = x.to("cpu")
                ek[i] = torch.sum((x-x_recon)**2)/torch.sum(x**2)
        energy = (1-np.mean(ek))*100
        print('Recovered energy %.2f' % (energy))
        return rec
    
    def correlation(self, dataset):
        ##  Compute correlation between latent variables
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self._device)
            _,_,_, z = self(batch)
            np.save('z.npy',z.cpu())
            corr = np.corrcoef(z.cpu(),rowvar=False)
        detR = np.linalg.det(corr)*100
        print('Orthogonality between modes %.2f' % (detR))
        return corr.reshape((self.lat_dim*self.lat_dim,))
    
    def modes(self):
        zmode = np.diag(np.ones((self.lat_dim,),dtype=float))
        zmodt = torch.tensor(zmode, dtype=torch.float32)
        zmodt = zmodt.to(self._device)
        modes = self.decoder(zmodt)
        mymod = np.zeros((self.nx*self.ny,self.lat_dim),dtype=float)
        modes = modes.cpu()
        for imode in range(self.lat_dim):
            modesr = modes[imode,0,:,:].detach().numpy()
            mymod[:,imode] = modesr.reshape((self.nx*self.ny,), order='C')
        return mymod.reshape((self.nx*self.ny*self.lat_dim,),order='C')

    def latent_space(self, dataset):
        # Compute latent vectors
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self._device)
            print(batch)
            _,_,_, z = self(batch)
        return z

    def decode(self, z):
        zt  = torch.tensor(z, dtype=torch.float32)
        var = self.decoder(zt)
        var = var.cpu()
        varr = np.zeros((self.nx*self.ny,var.shape[0]),dtype=float)
        for it in range(var.shape[0]):
            varaux = var[it,0,:,:].detach().numpy()
            varr[:,it] = varaux.reshape((self.nx*self.ny,), order='C')
        return varr 
