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
        
        self.conv1  = nn.Conv2d(in_channels=1, out_channels=self.channels*1<<0, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv1.weight)
        
        self.conv2  = nn.Conv2d(in_channels=self.channels, out_channels=self.channels*1<<1, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv2.weight)
        
        self.conv3  = nn.Conv2d(in_channels=self.channels*1<<1, out_channels=self.channels*1<<2, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv3.weight)
        
        self.conv4  = nn.Conv2d(in_channels=self.channels*1<<2, out_channels= self.channels*1<<3, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv4.weight)
        
        self.conv5  = nn.Conv2d(in_channels=self.channels*1<<3, out_channels=self.channels*1<<4, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv5.weight)
        
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(in_features = int((self.channels*1<<4)*(self._nx/(1<<self.nlayers))*self._ny/(1<<self.nlayers)), out_features=128)
        nn.init.xavier_uniform_(self.fc1.weight)
        
        self.mu = nn.Linear(in_features = 128, out_features = self._lat_dim)
        nn.init.xavier_uniform_(self.mu.weight)
        
        self.logvar = nn.Linear(in_features = 128, out_features = self._lat_dim)
        nn.init.xavier_uniform_(self.logvar.weight)
    
    def reset_parameters(self):
        for layer in self.modules():
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
        nn.init.xavier_uniform_(self.fc1.weight)
        self.batch6 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(in_features = 128, out_features = int((self.channels*1<<4)*self._nx/(1<<self.nlayers)*self._ny/(1<<self.nlayers)))
        nn.init.xavier_uniform_(self.fc2.weight)
        self.batch5 = nn.BatchNorm1d(int((self.channels*1<<4)*self._nx/(1<<self.nlayers)*self._ny/(1<<self.nlayers)))
        
        self.conv5 = nn.ConvTranspose2d(in_channels = self.channels*1<<4, out_channels = self.channels*1<<3, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.batch4 = nn.BatchNorm2d(self.channels*1<<3)
        
        self.conv4 = nn.ConvTranspose2d(in_channels = self.channels*1<<3, out_channels = self.channels*1<<2, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.batch3 = nn.BatchNorm2d(self.channels*1<<2)
        
        self.conv3 = nn.ConvTranspose2d(in_channels = self.channels*1<<2, out_channels = self.channels*2, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.batch2 = nn.BatchNorm2d(self.channels*2)
        
        self.conv2 = nn.ConvTranspose2d(in_channels = self.channels*2, out_channels = self.channels, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.batch1 = nn.BatchNorm2d(self.channels)
        
        self.conv1 = nn.ConvTranspose2d(in_channels = self.channels, out_channels = 1 , kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv1.weight)
        
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
    
    def modes(self):
        zmode = np.diag(np.ones((self.lat_dim,),dtype=float))
        zmodt = torch.tensor(zmode, dtype=torch.float32)
        modes = self(zmodt)
        mymod = np.zeros((self.nx*self.ny,self.lat_dim),dtype=float)
        for imode in range(self.lat_dim):
            modesr = modes[imode,0,:,:].detach().numpy()
            mymod[:,imode] = modesr.reshape((self.nx*self.ny,), order='C')
        return mymod.reshape((self.nx*self.ny*self.lat_dim,),order='C')


class VariationalAutoencoder(nn.Module):
    def __init__(self, channels, latent_dim, nx, ny, kernel_size, stride, padding):
        super(VariationalAutoencoder, self).__init__()
        self._lat_dim = np.int(latent_dim)
        self._nx      = nx
        self._ny      = ny
        self.encoder  = VariationalEncoder(channels, latent_dim, self._nx, self._ny, kernel_size, stride, padding)
        self.decoder  = VariationalDecoder(channels, latent_dim, self._nx, self._ny, kernel_size, stride, padding)
        summary(self, input_size=(1, self._nx, self._ny))

    @property
    def nx(self):
        return self._nx
    
    @property
    def ny(self):
        return self._ny
    
    @property
    def lat_dim(self):
        return self._lat_dim
    
    def reparamatrizate(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.rand_like(std)  #we create a normal distribution (0 ,1 ) with the dimensions of std        
        sample = mu + std*epsilon
        return  sample
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparamatrizate(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
      
    def lossfunc(self, x, recon_x):
        return F.mse_loss(recon_x.view(-1, self.nx*self.ny), x.view(-1, self.nx*self.ny),reduction='mean')
        
    def kld(self, mu, logvar):    
        return 0.5*torch.mean(1 + logvar - mu**2 - logvar.exp())
    
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
        return rec, (1-np.mean(ek))*100
    
    def correlation(self, dataset):
        ##  Compute reconstruction and its accuracy
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            _,_,_, z = self(batch)
            corr = np.corrcoef(z,rowvar=False)
        return corr.reshape((self.lat_dim*self.lat_dim,)), np.linalg.det(corr)*100
    
      
## Early stopper callback
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
       
    def early_stop(self, validation_loss, prev_train, train):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif prev_train < train:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False