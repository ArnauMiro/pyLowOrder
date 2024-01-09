import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## Encoder and decoder without a pooling operation
'''

class Encoder2D(nn.Module):
    def __init__(self, nlayers, latent_dim, nx, ny, input_channels, filter_channels, kernel_size, padding, activation_funcs, nlinear, batch_norm=True, stride=1, pool_kernel_size=2, pool_stride=2):
        super(Encoder2D, self).__init__()

        self.nlayers = nlayers
        self.filt_chan = np.int(filter_channels)
        self.in_chan = np.int(input_channels)
        self._lat_dim = np.int(latent_dim)
        self._nx = np.int(nx)
        self._ny = np.int(ny)
        self.funcs = activation_funcs
        self.nlinear = nlinear
        self.batch_norm = batch_norm

        # Create lists to hold the convolutional and MaxPool layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()  # New list for MaxPool layers
        self.norm_layers = nn.ModuleList()
        in_channels = self.in_chan  # Initial input channels

        for ilayer in range(self.nlayers):
            out_channels = self.filt_chan * (1 << ilayer)  # Compute output channels
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv_layers.append(conv_layer)
            self.pool_layers.append(nn.MaxPool2d(pool_kernel_size, pool_stride))  # Add a MaxPool layer
            if self.batch_norm:
                self.norm_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels  # Update in_channels for the next layer

        # Adjust the input size for the first fully connected layer according to the MaxPool layers
        reduced_nx = self._nx // (pool_stride ** self.nlayers)
        reduced_ny = self._ny // (pool_stride ** self.nlayers)
        fc_input_size = out_channels * reduced_nx * reduced_ny
        self.flat     = nn.Flatten()
        self.fc1 = nn.Linear(fc_input_size, self.nlinear)
        self.mu = nn.Linear(self.nlinear, self._lat_dim)
        self.logvar = nn.Linear(self.nlinear, self._lat_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):        
        out = x
        for ilayer, (conv_layer, pool_layer) in enumerate(zip(self.conv_layers, self.pool_layers)):
            out = conv_layer(out)
            out = self.funcs[ilayer](out)
            out = pool_layer(out)  # Applying the MaxPool layer
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
        out = self.funcs[ilayer+1](self.flat(out))
        out = self.funcs[ilayer+2](self.fc1(out))
        out = self.mu(out)
        #logvar = self.logvar(out)
        return out #mu, logvar
    
class Decoder2D(nn.Module):
    def __init__(self, nlayers, latent_dim, nx, ny, input_channels, filter_channels, kernel_size, padding, activation_funcs, nlinear, batch_norm=True, stride=1, upsample_mode='nearest'):
        super(Decoder2D, self).__init__()       
        self.nlayers = nlayers
        self.filt_chan = filter_channels
        self.in_chan = input_channels
        self.lat_dim = latent_dim
        self.nx = nx
        self.ny = ny
        self.funcs = activation_funcs
        self.nlinear = nlinear
        self.batch_norm = batch_norm

        self.fc1 = nn.Linear(in_features=self.lat_dim, out_features=self.nlinear)
        fc_output_size = (self.filt_chan * (1 << (self.nlayers-1)) * self.nx // (1 << self.nlayers) * self.ny // (1 << self.nlayers))
        self.fc2 = nn.Linear(in_features=self.nlinear, out_features=fc_output_size)

        # Create lists to hold the transposed convolutional and Upsample layers
        self.deconv_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()  # New list for Upsample layers
        self.norm_layers = nn.ModuleList()
        in_channels = self.filt_chan * (1 << (self.nlayers-1))

        for i in range(self.nlayers-1, 0, -1):
            out_channels = self.filt_chan * (1 << (i - 1))  # Compute output channels
            deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            self.deconv_layers.append(deconv_layer)
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))  # Add an Upsample layer
            if self.batch_norm:
                self.norm_layers.append(nn.BatchNorm2d(in_channels))
            in_channels = out_channels  # Update in_channels for the next layer

        out_channels = self.in_chan
        deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.deconv_layers.append(deconv_layer)
        self.upsample_layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))  # Add an Upsample layer
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
        out = out.view(out.size(0), self.filt_chan * (1 << (self.nlayers-1)), self.nx // (1 << self.nlayers), self.ny // (1 << self.nlayers))
        for ilayer, (deconv_layer, upsample_layer) in enumerate(zip(self.deconv_layers, self.upsample_layers)):
            out = upsample_layer(out)  # Applying the Upsample layer
            if self.batch_norm and ilayer < len(self.norm_layers):
                out = self.norm_layers[ilayer](out)
            out = self.funcs[self.nlayers-ilayer-1](deconv_layer(out))
        return out

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = self.e11(x)
        xe11 = torch.tanh(xe11)  # Apply Tanh activation function
        xe12 = self.e12(xe11)
        xe12 = torch.tanh(xe12)  # Apply Tanh activation function
        xp1 = self.pool1(xe12)

        xe21 = self.e21(xp1)
        xe21 = torch.tanh(xe21)  # Apply Tanh activation function
        xe22 = self.e22(xe21)
        xe22 = torch.tanh(xe22)  # Apply Tanh activation function
        xp2 = self.pool2(xe22)

        xe31 = self.e31(xp2)
        xe31 = torch.tanh(xe31)  # Apply Tanh activation function
        xe32 = self.e32(xe31)
        xe32 = torch.tanh(xe32)  # Apply Tanh activation function
        xp3 = self.pool3(xe32)

        xe41 = self.e41(xp3)
        xe41 = torch.tanh(xe41)  # Apply Tanh activation function
        xe42 = self.e42(xe41)
        xe42 = torch.tanh(xe42)  # Apply Tanh activation function
        xp4 = self.pool4(xe42)

        xe51 = self.e51(xp4)
        xe51 = torch.tanh(xe51)  # Apply Tanh activation function
        xe52 = self.e52(xe51)
        xe52 = torch.tanh(xe52)  # Apply Tanh activation function

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.d11(xu11)
        xd11 = torch.tanh(xd11)  # Apply Tanh activation function
        xd12 = self.d12(xd11)
        xd12 = torch.tanh(xd12)  # Apply Tanh activation function

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.d21(xu22)
        xd21 = torch.tanh(xd21)  # Apply Tanh activation function
        xd22 = self.d22(xd21)
        xd22 = torch.tanh(xd22)  # Apply Tanh activation function

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.d31(xu33)
        xd31 = torch.tanh(xd31)  # Apply Tanh activation function
        xd32 = self.d32(xd31)
        xd32 = torch.tanh(xd32)  # Apply Tanh activation function

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.d41(xu44)
        xd41 = torch.tanh(xd41)  # Apply Tanh activation function
        xd42 = self.d42(xd41)
        xd42 = torch.tanh(xd42)  # Apply Tanh activation function

        # Output layer
        out = self.outconv(xd42)

        return out



'''
class Encoder2D(nn.Module):
    def __init__(self, nlayers, latent_dim, nx, ny, input_channels, filter_channels, kernel_size, padding, activation_funcs, nlinear, batch_norm=True, stride=2, dropout=0):
        super(Encoder2D, self).__init__()

        self.nlayers    = nlayers
        self.filt_chan  = np.int(filter_channels)
        self.in_chan    = np.int(input_channels)
        self._lat_dim   = np.int(latent_dim)
        self._nx        = np.int(nx)
        self._ny        = np.int(ny)
        self.funcs      = activation_funcs
        self.nlinear    = nlinear
        self.batch_norm = batch_norm
        self.dropout    = nn.Dropout(p=dropout)

        # Create a list to hold the convolutional layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        in_channels = self.in_chan # Initial input channels
        for ilayer in range(self.nlayers):
            out_channels = self.filt_chan * (1 << ilayer)  # Compute output channels
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
            out = conv_layer(self.dropout(out))
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[ilayer](out)
        out = self.funcs[ilayer+1](self.flat(out))
        out = self.funcs[ilayer+2](self.fc1(out))
        mu = self.mu(out)
        logvar = self.logvar(out)
        return mu, logvar
    
class Decoder2D(nn.Module):
    def __init__(self, nlayers, latent_dim, nx, ny, input_channels, filter_channels, kernel_size, padding, activation_funcs, nlinear, batch_norm=True, stride=2, dropout=0):
        super(Decoder2D, self).__init__()       
        
        self.nlayers    = nlayers
        self.filt_chan  = filter_channels
        self.in_chan    = input_channels
        self.lat_dim    = latent_dim
        self.nx         = nx
        self.ny         = ny
        self.funcs      = activation_funcs
        self.nlinear    = nlinear
        self.batch_norm = batch_norm
        self.dropout    = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(in_features=self.lat_dim, out_features=self.nlinear)
        fc_output_size = int((self.filt_chan * (1 << (self.nlayers-1)) * self.nx // (1 << self.nlayers) * self.ny // (1 << self.nlayers)))
        self.fc2 = nn.Linear(in_features=self.nlinear, out_features=fc_output_size)

        # Create a list to hold the transposed convolutional layers
        self.deconv_layers = nn.ModuleList()
        self.norm_layers   = nn.ModuleList()
        in_channels = self.filt_chan * (1 << self.nlayers-1)
        for i in range(self.nlayers-1, 0, -1):
            out_channels = self.filt_chan * (1 << (i - 1))  # Compute output channels
            deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            self.deconv_layers.append(deconv_layer)
            if self.batch_norm:
                self.norm_layers.append(nn.BatchNorm2d(in_channels))
            in_channels = out_channels  # Update in_channels for the next layer
        out_channels = self.in_chan
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
        out = out.view(out.size(0), self.filt_chan * (1 << (self.nlayers-1)), int(self.nx // (1 << self.nlayers)), int(self.ny // (1 << self.nlayers)))
        for ilayer, (deconv_layer) in enumerate(self.deconv_layers):
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[self.nlayers-ilayer-1](deconv_layer(self.dropout(out)))
        return out
