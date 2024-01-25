import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np

#from   torch.utils.tensorboard import SummaryWriter
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
class Autoencoder(nn.Module):
    def __init__(self, latent_dim, nx, ny, input_channels, encoder, decoder, device='cpu'):
        super(Autoencoder, self).__init__()
        self.lat_dim  = np.int(latent_dim)
        self.nx       = nx
        self.ny       = ny
        self.inp_chan = input_channels
        self.encoder  = encoder
        self.decoder  = decoder
        self._device  = device
        encoder.to(self._device)
        decoder.to(self._device)
        self.to(self._device)
        summary(self, input_size=(self.inp_chan, self.nx, self.ny))
      
    def _lossfunc(self, x, recon_x, reduction):
        return  F.mse_loss(recon_x.view(-1, self.nx*self.ny), x.view(-1, self.nx*self.ny),reduction=reduction)
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z 
       
    def train_model(self, train_data, vali_data, nepochs, callback=None, learning_rate=6e-4, BASEDIR='./', reduction='mean', alpha=0.1):
        prev_train_loss = 1e99
        writer = SummaryWriter(BASEDIR)
        for epoch in range(nepochs):
            self.train()
            num_batches = 0 
            learning_rate = learning_rate * 1/(1 + 0.0001 * epoch)               #HARDCODED!!
            optimizer = torch.optim.AdamW(self.parameters(), lr= learning_rate)  #HARDCODED!!
            tr_loss = 0
            for batch in train_data:
                recon, _ = self(batch)
                loss     = self._lossfunc(batch, recon, reduction)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss     += loss.item()
                num_batches += 1
            with torch.no_grad():
                val_batches = 0
                va_loss     = 0
                for val_batch in vali_data:
                    val_recon,  _ = self(val_batch)
                    vali_loss     = self._lossfunc(val_batch, val_recon, reduction)
                    va_loss      += vali_loss.item()
                    val_batches  += 1
                tr_loss/=num_batches
                va_loss/=val_batches
                writer.add_scalar("Loss/train",tr_loss,epoch+1)
                writer.add_scalar("Loss/vali", va_loss,epoch+1)

            #if callback.early_stop(va_loss, prev_train_loss, tr_loss):
            #    print('Early Stopper Activated at epoch %i' %epoch, flush=True)
            #    break
            prev_train_loss = tr_loss   
            print('Epoch [%d / %d] average training loss: %.5e | average validation loss: %.5e' % (epoch+1, nepochs, tr_loss, va_loss), flush=True)
        writer.flush()
        writer.close()
        torch.save(self.state_dict(), '%s/model_state' % BASEDIR)

    def reconstruct(self, dataset):
        ##  Compute reconstruction and its accuracy
        ek     = np.zeros((len(dataset),))
        rec    = np.zeros((self.inp_chan,self.nx*self.ny,len(dataset))) 
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            ## Energy recovered in reconstruction
            instant = iter(loader)
            energy_batch = next(instant)
            for i in range(len(dataset)):
                x = energy_batch[i,:,:,:]
                x = torch.reshape(x, [1,self.inp_chan,self.nx,self.ny])
                x = x.to(self._device)
                x_recon  = self(x)
                x_recon  = np.asanyarray(x_recon[0].cpu())
                for ichan in range(self.inp_chan):
                    x_recchan  = x_recon[0,ichan,:,:]
                    x_recchan  = torch.reshape(torch.tensor(x_recchan),[self.nx*self.ny,])
                    rec[ichan,:,i] = x_recchan.detach().numpy()
                xr    = rec.reshape((self.inp_chan*self.nx*self.ny,len(dataset)))
                x     = torch.reshape(x,[self.inp_chan*self.nx*self.ny])
                x     = x.to("cpu")
                ek[i] = torch.sum((x-xr[:,i])**2)/torch.sum(x**2)
        energy = (1-np.mean(ek))*100
        print('Recovered energy %.2f' % (energy))
        return rec
    
    def latent_space(self, dataset):
        # Compute latent vectors
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self._device)
            _,z = self(batch)
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

## Wrapper of a variational autoencoder
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, nx, ny, input_channels, encoder, decoder, device='cpu'):
        super(VariationalAutoencoder, self).__init__()
        self.lat_dim  = np.int(latent_dim)
        self.nx       = nx
        self.ny       = ny
        self.inp_chan = input_channels
        self.encoder  = encoder
        self.decoder  = decoder
        self._device  = device
        encoder.to(self._device)
        decoder.to(self._device)
        self.to(self._device)
        summary(self, input_size=(self.inp_chan, self.nx, self.ny))
   
    def _reparamatrizate(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)  #we create a normal distribution (0 ,1 ) with the dimensions of std        
        sample = mu + std*epsilon
        return  sample
    
    def _lossfunc(self, x, recon_x, reduction='mean'):
        return  F.mse_loss(recon_x.view(-1, self.nx*self.ny), x.view(-1, self.nx*self.ny),reduction=reduction)
           
    def _kld(self, mu, logvar):    
        return 0.5*torch.sum(1 + logvar - mu**2 - logvar.exp())
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._reparamatrizate(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
       
    def train_model(self, train_data, vali_data, beta, nepochs, callback=None, learning_rate=5e-4, BASEDIR='./'):
        prev_train_loss = 1e99
        #writer = SummaryWriter(BASEDIR)
        prevloss    = 0
        for epoch in range(nepochs):
            self.train()
            num_batches = 0 
            learning_rate = learning_rate * 1/(1 + 0.0001 * epoch)               #HARDCODED!!
            optimizer = torch.optim.AdamW(self.parameters(), lr= learning_rate)  #HARDCODED!!
            tr_loss = 0
            mse     = 0
            kld     = 0
            for batch in train_data:
                recon, mu, logvar, _ = self(batch)
                mse_i  = self._lossfunc(batch, recon, reduction='mean')
                bkld_i = self._kld(mu,logvar)*beta
                loss = mse_i - bkld_i
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Epoch [%d / %d] batch loss: %.5e' % (epoch+1, nepochs, loss.item()), flush=True)
                tr_loss += loss.item()
                mse     += self._lossfunc(batch, recon).item()
                kld     += self._kld(mu,logvar).item()*beta
                num_batches += 1
            with torch.no_grad():
                val_batches = 0
                va_loss     = 0
                for val_batch in vali_data:
                    val_recon, val_mu, val_logvar, _ = self(val_batch)
                    mse_i       = self._lossfunc(val_batch, val_recon, reduction='mean')
                    bkld_i      = self._kld(val_mu,val_logvar)*beta
                    vali_loss   = mse_i - bkld_i
                    va_loss     += vali_loss.item()
                    val_batches += 1
                tr_loss/=num_batches
                va_loss/=val_batches
                mse /= num_batches
                kld /= num_batches
                #writer.add_scalar("Loss/train",tr_loss,epoch+1)
                #writer.add_scalar("Loss/vali", va_loss,epoch+1)
                #writer.add_scalar("Loss/mse",  mse,    epoch+1)
                #writer.add_scalar("Loss/kld",  kld,    epoch+1)

            #if callback.early_stop(va_loss, prev_train_loss, tr_loss):
            #    print('Early Stopper Activated at epoch %i' %epoch, flush=True)
            #    break
            prev_train_loss = tr_loss   
            print('Epoch [%d / %d] average training loss: %.5e | average validation loss: %.5e' % (epoch+1, nepochs, tr_loss, va_loss), flush=True)
        #qwriter.flush()
        #qwriter.close()
        torch.save(self.state_dict(), '%s/model_state' % BASEDIR)

    def reconstruct(self, dataset):
        ##  Compute reconstruction and its accuracy
        ek     = np.zeros((len(dataset),))
        rec    = np.zeros((self.inp_chan,self.nx*self.ny,len(dataset))) 
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            ## Energy recovered in reconstruction
            instant = iter(loader)
            energy_batch = next(instant)
            for i in range(len(dataset)):
                x = energy_batch[i,:,:,:]
                x = torch.reshape(x, [1,self.inp_chan,self.nx,self.ny])
                x = x.to(self._device)
                x_recon  = self(x)
                x_recon  = np.asanyarray(x_recon[0].cpu())
                for ichan in range(self.inp_chan):
                    x_recchan  = x_recon[0,ichan,:,:]
                    x_recchan  = torch.reshape(torch.tensor(x_recchan),[self.nx*self.ny,])
                    rec[ichan,:,i] = x_recchan.detach().numpy()
                xr    = rec.reshape((self.inp_chan*self.nx*self.ny,len(dataset)))
                x     = torch.reshape(x,[self.inp_chan*self.nx*self.ny])
                x     = x.to("cpu")
                ek[i] = torch.sum((x-xr[:,i])**2)/torch.sum(x**2)
        energy = (1-np.mean(ek))*100
        print('Recovered energy %.2f' % (energy))
        return rec, ek
    
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
        return corr, detR#.reshape((self.lat_dim*self.lat_dim,))
    
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
