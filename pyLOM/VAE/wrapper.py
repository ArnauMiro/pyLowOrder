import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from   torchsummary import summary


## Definition of a variational autoencoder
class VariationalEncoder(nn.Module):
    def __init__(self, channels, latent_dim, nx, ny, kernel_size, stride, padding):
        super(VariationalEncoder, self).__init__()

        self.nlayers  = 5   
        self.channels = np.int(channels)
        self._lat_dim = np.int(latent_dim)
        self._nx      = np.int(nx)
        self._ny      = np.int(ny)

        self.drop1 = nn.Dropout1d()
        self.drop2 = nn.Dropout2d()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channels*1<<0, kernel_size=kernel_size, stride=stride, padding=padding)       
        self.conv2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels*1<<1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=self.channels*1<<1, out_channels=self.channels*1<<2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=self.channels*1<<2, out_channels= self.channels*1<<3, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv5 = nn.Conv2d(in_channels=self.channels*1<<3, out_channels=self.channels*1<<4, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.flat   = nn.Flatten()
        self.fc1    = nn.Linear(in_features = int((self.channels*1<<4)*(self._nx/(1<<self.nlayers))*self._ny/(1<<self.nlayers)), out_features=128)       
        self.mu     = nn.Linear(in_features = 128, out_features = self._lat_dim)
        self.logvar = nn.Linear(in_features = 128, out_features = self._lat_dim)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):        
        out    = torch.tanh(self.conv1(x))
        out    = torch.nn.functional.elu(self.conv2(out))
        out    = torch.nn.functional.elu(self.conv3(out))
        out    = torch.nn.functional.elu(self.conv4(out))
        out    = torch.nn.functional.elu(self.conv5(out))
        out    = self.flat(out)
        out    = torch.nn.functional.elu(self.fc1(out))
        mu     = self.mu(out)
        logvar = self.logvar(out)
        return  mu, logvar
    
class VariationalDecoder(nn.Module):
    def __init__(self, channels, latent_dim, nx, ny, kernel_size, stride, padding):
        super(VariationalDecoder , self).__init__()       
        
        self.nlayers   = 5
        self.channels  = channels
        self._lat_dim  = latent_dim
        self._nx       = nx
        self._ny       = ny

        self.drop2 = nn.Dropout2d()
        self.drop1 = nn.Dropout1d()
        
        self.fc1 = nn.Linear(in_features = self._lat_dim, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = int((self.channels*1<<4)*self._nx/(1<<self.nlayers)*self._ny/(1<<self.nlayers)))
        
        self.conv5 = nn.ConvTranspose2d(in_channels = self.channels*1<<4, out_channels = self.channels*1<<3, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.ConvTranspose2d(in_channels = self.channels*1<<3, out_channels = self.channels*1<<2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.ConvTranspose2d(in_channels = self.channels*1<<2, out_channels = self.channels*2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.ConvTranspose2d(in_channels = self.channels*2, out_channels = self.channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1 = nn.ConvTranspose2d(in_channels = self.channels, out_channels = 1 , kernel_size=kernel_size, stride=stride, padding=padding)

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
    @property
    def nx(self):
        return self._nx
    
    @property
    def ny(self):
        return self._ny
    
    @property
    def lat_dim(self):
        return self._lat_dim

    def forward(self, x):
        out = torch.nn.functional.elu(self.fc1(x))      
        out = torch.nn.functional.elu(self.fc2(out))
        out = out.view(out.size(0),self.channels*1<<4, int(self._nx/(1<<self.nlayers)), int(self._ny/(1<<self.nlayers)))
        out = torch.nn.functional.elu(self.conv5(out))
        out = torch.nn.functional.elu(self.conv4(out))
        out = torch.nn.functional.elu(self.conv3(out))
        out = torch.nn.functional.elu(self.conv2(out))
        out = torch.tanh(self.conv1(out))
        return out

class VariationalAutoencoder(nn.Module):
    def __init__(self, channels, latent_dim, nx, ny, kernel_size, stride, padding):
        super(VariationalAutoencoder, self).__init__()
        self.lat_dim = np.int(latent_dim)
        self.nx      = nx
        self.ny      = ny
        self.encoder  = VariationalEncoder(channels, latent_dim, self.nx, self.ny, kernel_size, stride, padding)
        self.decoder  = VariationalDecoder(channels, latent_dim, self.nx, self.ny, kernel_size, stride, padding)
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
       
    def train_model(self, train_data, vali_data, beta, nepochs, callback=None, learning_rate=3e-4):
        self.train()
        prev_train_loss = 1e99
        train_loss_avg  = [] #Build numpy array as nepoch*num_batches
        val_loss        = [] #Build numpy array as nepoch*num_batches*vali_batches
        mse             = [] #Build numpy array as nepoch*num_batches
        kld             = [] #Build numpy array as nepoch*num_batches
        for epoch in range(nepochs):
            mse.append(0)
            kld.append(0)
            train_loss_avg.append(0)
            num_batches = 0 
            learning_rate = learning_rate * 1/(1 + 0.001 * epoch)               #HARDCODED!!
            optimizer = torch.optim.Adam(self.parameters(), lr= learning_rate)  #HARDCODED!!
            for batch in train_data:     
                recon, mu, logvar, _ = self(batch)
                mse_i  = self._lossfunc(batch, recon)
                bkld_i = self._kld(mu,logvar)*beta
                loss   = mse_i - bkld_i
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_avg[-1] += loss.item()
                mse[-1] = self._lossfunc(batch, recon).item()
                kld[-1] = self._kld(mu,logvar).item()*beta
                num_batches += 1
            with torch.no_grad():
                val_batches = 0
                val_loss.append(0)
                for val_batch in vali_data:
                    val_recon, val_mu, val_logvar , _ = self(val_batch)
                    mse_i     = self._lossfunc(val_batch, val_recon)
                    bkld_i    = self._kld(val_mu,val_logvar)*beta
                    vali_loss = mse_i - bkld_i
                    val_loss[-1] += vali_loss.item()
                    val_batches += 1
                val_loss[-1] /= num_batches
                mse[-1]      /= num_batches
                kld[-1]      /= num_batches
                train_loss_avg[-1] /= num_batches
            if callback.early_stop(val_loss[-1], prev_train_loss, train_loss_avg[-1]):
                print('Early Stopper Activated at epoch %i' %epoch)
                break
            prev_train_loss = train_loss_avg[-1]   
            print('Epoch [%d / %d] average training error: %.5e' % (epoch+1, nepochs, train_loss_avg[-1]))
            
        return np.array(train_loss_avg), np.array(val_loss), np.array(mse), np.array(kld)
    
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
                x_recon  = self(x)
                x_recon  = np.asanyarray(x_recon[0])
                x_recon  = x_recon[0,0,:,:]
                x_recon  = torch.reshape(torch.tensor(x_recon),[self.nx*self.ny, 1])
                rec[:,i] = x_recon.detach().numpy()[:,0]
                x = torch.reshape(x,[self.nx*self.ny,1])
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
            _,_,_, z = self(batch)
            corr = np.corrcoef(z,rowvar=False)
        detR = np.linalg.det(corr)*100
        print('Correlation between modes %.2f' % (detR))
        return corr.reshape((self.lat_dim*self.lat_dim,))
    
    def modes(self):
        zmode = np.diag(np.ones((self.lat_dim,),dtype=float))
        zmodt = torch.tensor(zmode, dtype=torch.float32)
        modes = self.decoder(zmodt)
        mymod = np.zeros((self.nx*self.ny,self.lat_dim),dtype=float)
        for imode in range(self.lat_dim):
            modesr = modes[imode,0,:,:].detach().numpy()
            mymod[:,imode] = modesr.reshape((self.nx*self.ny,), order='C')
        return mymod.reshape((self.nx*self.ny*self.lat_dim,),order='C')