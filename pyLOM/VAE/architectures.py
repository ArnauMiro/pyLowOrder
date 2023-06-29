import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## Encoder and decoder without a pooling operation
class EncoderNoPool(nn.Module):
    def __init__(self, latent_dim, nx, ny, channels, kernel_size, padding, stride=2):
        super(EncoderNoPool, self).__init__()

        self.nlayers  = 5   
        self.channels = np.int(channels)
        self._lat_dim = np.int(latent_dim)
        self._nx      = np.int(nx)
        self._ny      = np.int(ny)

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
    
class DecoderNoPool(nn.Module):
    def __init__(self, latent_dim, nx, ny, channels, kernel_size, padding, stride=2):
        super(DecoderNoPool, self).__init__()       
        
        self.nlayers  = 5
        self.channels = channels
        self.lat_dim  = latent_dim
        self.nx       = nx
        self.ny       = ny

        self.fc1 = nn.Linear(in_features = self.lat_dim, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = int((self.channels*1<<4)*self.nx/(1<<self.nlayers)*self.ny/(1<<self.nlayers)))
        
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

    def forward(self, x):
        out = torch.nn.functional.elu(self.fc1(x))      
        out = torch.nn.functional.elu(self.fc2(out))
        out = out.view(out.size(0),self.channels*1<<4, int(self.nx/(1<<self.nlayers)), int(self.ny/(1<<self.nlayers)))
        out = torch.nn.functional.elu(self.conv5(out))
        out = torch.nn.functional.elu(self.conv4(out))
        out = torch.nn.functional.elu(self.conv3(out))
        out = torch.nn.functional.elu(self.conv2(out))
        out = torch.tanh(self.conv1(out))
        return out

## Encoder and decoder with a max pool operation
class EncoderMaxPool(nn.Module):
    def __init__(self, latent_dim, nx, ny, channels, kernel_size, padding, stride=2):
        super(EncoderMaxPool, self).__init__()

        self.nlayers  = 5   
        self.channels = np.int(channels)
        self._lat_dim = np.int(latent_dim)
        self._nx      = np.int(nx)
        self._ny      = np.int(ny)

        self.conv1 = nn.Conv2d(in_channels=1,                  out_channels=self.channels*1<<0, kernel_size=kernel_size, stride=1, padding=padding)       
        self.conv2 = nn.Conv2d(in_channels=self.channels,      out_channels=self.channels*1<<1, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=self.channels*1<<1, out_channels=self.channels*1<<2, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=self.channels*1<<2, out_channels=self.channels*1<<3, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv5 = nn.Conv2d(in_channels=self.channels*1<<3, out_channels=self.channels*1<<4, kernel_size=kernel_size, stride=1, padding=padding)
        
        self.pool = nn.MaxPool2d(kernel_size=stride)

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
        out = torch.tanh(self.conv1(x))
        out = self.pool(out)        
        out = torch.tanh(self.conv2(out))
        out = self.pool(out)
        out = torch.tanh(self.conv3(out))
        out = self.pool(out)
        out = torch.tanh(self.conv4(out))
        out = self.pool(out)
        out = torch.tanh(self.conv5(out))
        out = self.pool(out)
        out = self.flat(out)
        out = torch.tanh(self.fc1(out))
        mu = self.mu(out)
        logvar = self.logvar(out)
        return mu, logvar
    
class DecoderMaxPool(nn.Module):
    def __init__(self, latent_dim, nx, ny, channels, kernel_size, padding, stride=2):
        super(DecoderMaxPool, self).__init__()       
        
        self.nlayers  = 5
        self.channels = channels
        self.lat_dim  = latent_dim
        self.nx       = nx
        self.ny       = ny
        
        self.fc1 = nn.Linear(in_features = self.lat_dim, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = int((self.channels*1<<4)*self.nx/(1<<self.nlayers)*self.ny/(1<<self.nlayers)))
        
        self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')

        self.conv5 = nn.ConvTranspose2d(in_channels=self.channels*1<<4, out_channels=self.channels*1<<3, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv4 = nn.ConvTranspose2d(in_channels=self.channels*1<<3, out_channels=self.channels*1<<2, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.channels*1<<2, out_channels=self.channels*2,    kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.channels*2,    out_channels=self.channels,      kernel_size=kernel_size, stride=1, padding=padding)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.channels,      out_channels=1,                  kernel_size=kernel_size, stride=1, padding=padding)

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        out = out.view(out.size(0), self.channels*1 << 4, int(self.nx/(1 << self.nlayers)), int(self.ny/(1 << self.nlayers)))
        out = self.upsample(out)
        out = torch.tanh(self.conv5(out))
        out = self.upsample(out)  
        out = torch.tanh(self.conv4(out))
        out = self.upsample(out)  
        out = torch.tanh(self.conv3(out))
        out = self.upsample(out) 
        out = torch.tanh(self.conv2(out))
        out = self.upsample(out) 
        out = torch.tanh(self.conv1(out))
        return out