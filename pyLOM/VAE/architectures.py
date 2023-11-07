import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## Encoder and decoder without a pooling operation
class EncoderNoPool(nn.Module):
    def __init__(self, nlayers, latent_dim, nx, ny, channels, kernel_size, padding, activation_funcs, nlinear, batch_norm=True, stride=2):
        super(EncoderNoPool, self).__init__()

        self.nlayers    = nlayers
        self.channels   = np.int(channels)
        self._lat_dim   = np.int(latent_dim)
        self._nx        = np.int(nx)
        self._ny        = np.int(ny)
        self.funcs      = activation_funcs
        self.nlinear    = nlinear
        self.batch_norm = batch_norm

        # Create a list to hold the convolutional layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        in_channels = 1  # Initial input channels
        for ilayer in range(self.nlayers):
            out_channels = self.channels * (1 << ilayer)  # Compute output channels
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv_layers.append(conv_layer)
            if self.batch_norm:
                self.norm_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels  # Update in_channels for the next layer
       
        self.flat     = nn.Flatten()
        fc_input_size = out_channels * (self._nx // (1 << self.nlayers)) * (self._ny // (1 << self.nlayers))
        self.fc1      = nn.Linear(fc_input_size, self.nlinear)
        self.mu       = nn.Linear(self.nlinear, self._lat_dim)
        self.logvar   = nn.Linear(self.nlinear, self._lat_dim)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):        
        out = x
        for ilayer, conv_layer in enumerate(self.conv_layers):
            out = conv_layer(out)
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[ilayer](out)
        out = self.funcs[ilayer+1](self.flat(out))
        out = self.funcs[ilayer+2](self.fc1(out))
        mu = self.mu(out)
        logvar = self.logvar(out)
        return mu, logvar
    
class DecoderNoPool(nn.Module):
    def __init__(self, nlayers, latent_dim, nx, ny, channels, kernel_size, padding, activation_funcs, nlinear, batch_norm=True, stride=2):
        super(DecoderNoPool, self).__init__()       
        
        self.nlayers    = nlayers
        self.channels   = channels
        self.lat_dim    = latent_dim
        self.nx         = nx
        self.ny         = ny
        self.funcs      = activation_funcs
        self.nlinear    = nlinear
        self.batch_norm = batch_norm

        self.fc1 = nn.Linear(in_features=self.lat_dim, out_features=self.nlinear)
        fc_output_size = int((self.channels * (1 << (self.nlayers-1)) * self.nx // (1 << self.nlayers) * self.ny // (1 << self.nlayers)))
        self.fc2 = nn.Linear(in_features=self.nlinear, out_features=fc_output_size)

        # Create a list to hold the transposed convolutional layers
        self.deconv_layers = nn.ModuleList()
        self.norm_layers   = nn.ModuleList()
        in_channels = self.channels * (1 << self.nlayers-1)
        for i in range(self.nlayers-1, 0, -1):
            out_channels = self.channels * (1 << (i - 1))  # Compute output channels
            deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            self.deconv_layers.append(deconv_layer)
            if self.batch_norm:
                self.norm_layers.append(nn.BatchNorm2d(in_channels))
            in_channels = out_channels  # Update in_channels for the next layer
        out_channels = 1
        deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.deconv_layers.append(deconv_layer)
        if self.batch_norm:
            self.norm_layers.append(nn.BatchNorm2d(in_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.funcs[self.nlayers+1](self.fc1(x))
        out = self.funcs[self.nlayers](self.fc2(out))
        out = out.view(out.size(0), self.channels * (1 << (self.nlayers-1)), int(self.nx // (1 << self.nlayers)), int(self.ny // (1 << self.nlayers)))
        for ilayer, (deconv_layer) in enumerate(self.deconv_layers):
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[self.nlayers-ilayer-1](deconv_layer(out))
        #out = torch.tanh(out)
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

        self.bn1   = nn.BatchNorm2d(self.channels*1<<0)       
        self.bn2   = nn.BatchNorm2d(self.channels*1<<1)       
        self.bn3   = nn.BatchNorm2d(self.channels*1<<2)       
        self.bn4   = nn.BatchNorm2d(self.channels*1<<3)       
        self.bn5   = nn.BatchNorm2d(self.channels*1<<4)       
        
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
        out = torch.tanh(self.bn1(self.conv1(x)))
        out = self.pool(out)   
        out = torch.tanh(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = torch.tanh(self.bn3(self.conv3(out)))
        out = self.pool(out)
        out = torch.tanh(self.bn4(self.conv4(out)))
        out = self.pool(out)
        out = torch.tanh(self.bn5(self.conv5(out)))
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc1(out)
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

        self.bn1   = nn.BatchNorm2d(self.channels*1<<0)       
        self.bn2   = nn.BatchNorm2d(self.channels*1<<1)       
        self.bn3   = nn.BatchNorm2d(self.channels*1<<2)       
        self.bn4   = nn.BatchNorm2d(self.channels*1<<3)       
        self.bn5   = nn.BatchNorm2d(self.channels*1<<4)  

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = out.view(out.size(0), self.channels*1 << 4, int(self.nx/(1 << self.nlayers)), int(self.ny/(1 << self.nlayers)))
        out = self.bn5(self.upsample(out))
        out = torch.tanh(self.bn4(self.conv5(out)))
        out = self.upsample(out)  
        out = torch.tanh(self.bn3(self.conv4(out)))
        out = self.upsample(out)  
        out = torch.tanh(self.bn2(self.conv3(out)))
        out = self.upsample(out) 
        out = torch.tanh(self.bn1(self.conv2(out)))
        out = self.upsample(out) 
        out = torch.tanh(self.conv1(out))
        return out
