import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np

from   torch.cuda.amp            import GradScaler, autocast
from   torch.utils.tensorboard import SummaryWriter
from   torchsummary            import summary

from   ..utils.cr              import cr


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
        self.lat_dim  = latent_dim
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
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon, z  

    def train_model(self, train_data, vali_data, nepochs, callback=None, learning_rate=1e-3, BASEDIR='./', reduction='mean', lr_decay=0.999):
        # Initialization
        prev_train_loss = 1e99
        writer = SummaryWriter(BASEDIR)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        # Training loop
        for epoch in range(nepochs):
            self.train()
            num_batches = 0
            tr_loss = 0
            for batch in train_data:
                recon, _ = self(batch)
                loss = self._lossfunc(batch, recon, reduction)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                num_batches += 1
            tr_loss /= num_batches
            # Validation phase
            with torch.no_grad():
                val_batches = 0
                va_loss = 0
                for val_batch in vali_data:
                    val_recon, _ = self(val_batch)
                    vali_loss = self._lossfunc(val_batch, val_recon, reduction)
                    va_loss += vali_loss.item()
                    val_batches += 1
                va_loss /= val_batches
            # Logging
            writer.add_scalar("Loss/train", tr_loss, epoch + 1)
            writer.add_scalar("Loss/vali", va_loss, epoch + 1)
            # Early stopping
            if callback and callback.early_stop(va_loss, prev_train_loss, tr_loss):
                print(f'Early Stopper Activated at epoch {epoch}', flush=True)
                break
            prev_train_loss = tr_loss
            print(f'Epoch [{epoch+1} / {nepochs}] average training loss: {tr_loss:.5e} | average validation loss: {va_loss:.5e}', flush=True)            
            # Learning rate scheduling
            scheduler.step()

        # Cleanup
        writer.flush()
        writer.close()
        torch.save(self.state_dict(), f'{BASEDIR}/model_state.pth')

    def reconstruct(self, dataset):
        ##  Compute reconstruction and its accuracy
        ek     = np.zeros((len(dataset),))
        mean   = np.zeros((len(dataset),))
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
                xr      = rec.reshape((self.inp_chan*self.nx*self.ny,len(dataset)))
                x       = torch.reshape(x,[self.inp_chan*self.nx*self.ny])
                x       = x.to("cpu")
                ek[i]   = torch.sum((x-xr[:,i])**2)/torch.sum(x**2)
                mean[i] = torch.mean(torch.abs((x-xr[:,i])/x))
        energy = (1-np.mean(ek))*100
        print('Recovered energy %.2f' % (energy))
        print('Mean error %.2f percent' % (np.mean(mean)*100))
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
        self.lat_dim  = latent_dim
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
    

    @cr('VAE.train')   
    def train_model(self, train_data, vali_data, beta, nepochs, callback=None, learning_rate=5e-4, lr_decay=0.999, BASEDIR='./'):
        prev_train_loss = 1e99
        writer    = SummaryWriter(BASEDIR)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        scaler    = GradScaler()
        for epoch in range(nepochs):
            ## Training
            self.train()
            tr_loss = 0
            mse     = 0
            kld     = 0
            for batch in train_data:
                optimizer.zero_grad()
                with autocast():
                    recon, mu, logvar, _ = self(batch)
                    mse_i = self._lossfunc(batch, recon, reduction='mean')
                    kld_i = self._kld(mu,logvar)
                    loss  = mse_i - beta*kld_i
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tr_loss += loss.item()
                mse     += mse_i.item()
                kld     += kld_i.item()
            num_batches = len(train_data)
            tr_loss /= num_batches
            mse /= num_batches
            kld /= num_batches

            ## Validation
            self.eval()
            va_loss     = 0
            with torch.no_grad():
                for val_batch in vali_data:
                    with autocast():
                        val_recon, val_mu, val_logvar, _ = self(val_batch)
                        mse_i     = self._lossfunc(val_batch, val_recon, reduction='mean')
                        kld_i     = self._kld(val_mu,val_logvar)
                        vali_loss = mse_i - beta*kld_i
                    va_loss  += vali_loss.item()

            num_batches = len(vali_data)
            va_loss    /=num_batches
            writer.add_scalar("Loss/train",tr_loss,epoch+1)
            writer.add_scalar("Loss/vali", va_loss,epoch+1)
            writer.add_scalar("Loss/mse",  mse,    epoch+1)
            writer.add_scalar("Loss/kld",  kld,    epoch+1)

            if callback.early_stop(va_loss, prev_train_loss, tr_loss):
                print('Early Stopper Activated at epoch %i' %epoch, flush=True)
                break
            prev_train_loss = tr_loss   
            print('Epoch [%d / %d] average training loss: %.5e | average validation loss: %.5e' % (epoch+1, nepochs, tr_loss, va_loss), flush=True)
            # Learning rate scheduling
            scheduler.step()

        writer.flush()
        writer.close()
        torch.save(self.state_dict(), '%s/model_state' % BASEDIR)

    @cr('VAE.reconstruct')
    def reconstruct(self, dataset):
        ## Compute reconstruction and its accuracy
        num_samples = len(dataset)
        ek = np.zeros(num_samples)
        mu = np.zeros(num_samples)
        si = np.zeros(num_samples)
        rec = torch.zeros((self.inp_chan, self.nx * self.ny, num_samples), device=self._device)

        loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)

        with torch.no_grad():
            ## Energy recovered in reconstruction
            for energy_batch in loader:
                energy_batch = energy_batch.to(self._device)
                x_recon,_,_,_ = self(energy_batch)

                for i in range(num_samples):
                    x_recchan = x_recon[i]
                    rec[:, :, i] = x_recchan.view(self.inp_chan, self.nx * self.ny)

                    x = energy_batch[i].view(self.inp_chan * self.nx * self.ny)
                    xr = rec[:, :, i].view(self.inp_chan * self.nx * self.ny)

                    ek[i] = torch.sum((x - xr) ** 2) / torch.sum(x ** 2)
                    mu[i] = 2 * torch.mean(x) * torch.mean(xr) / (torch.mean(x) ** 2 + torch.mean(xr) ** 2)
                    si[i] = 2 * torch.std(x) * torch.std(xr) / (torch.std(x) ** 2 + torch.std(xr) ** 2)

        energy = (1 - np.mean(ek)) * 100
        print('Recovered energy %.2f' % energy)
        print('Recovered mean %.2f' % (np.mean(mu) * 100))
        print('Recovered fluct %.2f' % (np.mean(si) * 100))

        return rec.cpu().numpy()
  
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
