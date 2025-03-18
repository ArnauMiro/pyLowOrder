#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Encoder-Decoder architecture for NN Module
#
# Last rev: 09/10/2024

import torch.nn as nn


class Encoder2D(nn.Module):
    r"""
    Encoder2D class for the 2D Convolutional Autoencoder.

    Args:
        nlayers (int): Number of layers in the encoder.
        latent_dim (int): Latent dimension of the encoder.
        nh (int): Height of the input mesh/image.
        nw (int): Width of the input mesh/image.
        input_channels (int): Number of input channels.
        filter_channels (int): Number of filter channels.
        kernel_size (int): Kernel size for the convolutional layers.
        padding (int): Padding for the convolutional layers.
        activation_funcs (list): List of activation functions.
        nlinear (int): Number of neurons in the linear layer.
        batch_norm (bool): Whether to use batch normalization. Default is ``False``.
        stride (int): Stride for the convolutional layers. Default is ``2``.
        dropout (float): Dropout probability. Default is ``0``.
        vae (bool): Wheather the encoder is going to be used on a VAE or not. Default is ``False``.
    """
    def __init__(
        self,
        nlayers: int,
        latent_dim: int,
        nh: int,
        nw: int,
        input_channels: int,
        filter_channels: int,
        kernel_size: int,
        padding: int,
        activation_funcs: list,
        nlinear: int,
        batch_norm: bool = False,
        stride: int = 2,
        dropout: float = 0,
        vae: bool = False,
    ):
        super(Encoder2D, self).__init__()

        self.nlayers    = nlayers
        self.filt_chan  = filter_channels
        self.in_chan    = input_channels
        self.lat_dim    = latent_dim
        self.nh         = nh
        self.nw         = nw
        self.isvae      = vae
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
        fc_input_size = out_channels * (self.nh // (1 << self.nlayers)) * (self.nw // (1 << self.nlayers))
        self.fc1      = nn.Linear(fc_input_size, self.nlinear)
        if self.isvae:
            self.mu     = nn.Linear(self.nlinear, self.lat_dim)
            self.logvar = nn.Linear(self.nlinear, self.lat_dim)
        else:
            self.z = nn.Linear(self.nlinear, self.lat_dim)

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
        if self.isvae:
            return self.mu(out), self.logvar(out)
        else:
            return self.z(out)


class Decoder2D(nn.Module):
    r"""
    Decoder2D class for the 2D Convolutional Autoencoder.

    Args:
        nlayers (int): Number of layers in the encoder.
        latent_dim (int): Latent dimension of the encoder.
        nh (int): Height of the input mesh/image.
        nw (int): Width of the input mesh/image.
        input_channels (int): Number of input channels.
        filter_channels (int): Number of filter channels.
        kernel_size (int): Kernel size for the convolutional layers.
        padding (int): Padding for the convolutional layers.
        activation_funcs (list): List of activation functions.
        nlinear (int): Number of neurons in the linear layer.
        batch_norm (bool): Whether to use batch normalization. Default is ``False``.
        stride (int): Stride for the convolutional layers. Default is ``2``.
        dropout (float): Dropout probability. Default is ``0``.
    """
    def __init__(
        self,
        nlayers: int,
        latent_dim: int,
        nh: int,
        nw: int,
        input_channels: int,
        filter_channels: int,
        kernel_size: int,
        padding: int,
        activation_funcs: list,
        nlinear: int,
        batch_norm: bool = False,
        stride: int = 2,
        dropout: float = 0,
    ):
        super(Decoder2D, self).__init__()       
        
        self.nlayers    = nlayers
        self.filt_chan  = filter_channels
        self.in_chan    = input_channels
        self.lat_dim    = latent_dim
        self.nh         = nh
        self.nw         = nw
        self.funcs      = activation_funcs
        self.nlinear    = nlinear
        self.batch_norm = batch_norm
        self.dropout    = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(in_features=self.lat_dim, out_features=self.nlinear)
        fc_output_size = int((self.filt_chan * (1 << (self.nlayers-1)) * self.nh // (1 << self.nlayers) * self.nw // (1 << self.nlayers)))
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
        out = out.view(out.size(0), self.filt_chan * (1 << (self.nlayers-1)), int(self.nh // (1 << self.nlayers)), int(self.nw // (1 << self.nlayers)))
        for ilayer, (deconv_layer) in enumerate(self.deconv_layers[:-1]):
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[self.nlayers-ilayer-1](deconv_layer(out))
        return self.deconv_layers[-1](out)


class Encoder3D(nn.Module):
    r"""
    Encoder3D class for the 3D Convolutional Encoder.
    
    Args:
        nlayers (int): Number of layers in the encoder.
        latent_dim (int): Latent dimension of the encoder.
        nx (int): Height of the input mesh/image.
        ny (int): Width of the input mesh/image.
        nz (int): Depth of the input mesh/image.
        input_channels (int): Number of input channels.
        filter_channels (int): Number of filter channels.
        kernel_size (int): Kernel size for the convolutional layers.
        padding (int): Padding for the convolutional layers.
        activation_funcs (list): List of activation functions.
        nlinear (int): Number of neurons in the linear layer.
        batch_norm (bool): Whether to use batch normalization. Default is ``False``.
        stride (int): Stride for the convolutional layers. Default is ``2``.
        dropout (float): Dropout probability. Default is ``0``.
        vae (bool): Wheather the encoder is going to be used on a VAE or not. Default is ``False``.
    """
    def __init__(
        self,
        nlayers: int,
        latent_dim: int,
        nx: int,
        ny: int,
        nz: int,
        input_channels: int,
        filter_channels: int,
        kernel_size: int,
        padding: int,
        activation_funcs: list,
        nlinear: int,
        batch_norm: bool = False,
        stride: int = 2,
        dropout: float = 0,
        vae: bool = False,
    ):
        super(Encoder3D,self).__init__()

        self.nlayers = nlayers
        self.filt_chan = filter_channels
        self.in_chan = input_channels
        self._lat_dim = latent_dim
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._isvae = vae
        self.funcs = activation_funcs
        self.nlinear = nlinear
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(p=dropout)

        # List to hold the Conv Layers

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        in_channels = self.in_chan #Initial input channels
        for ilayer in range(self.nlayers):
            out_channels = self.filt_chan * (1 << ilayer) #Compute output channels using the shift operator
            conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv_layers.append(conv_layer)
            if self.batch_norm:
                self.norm_layers.append(nn.BatchNorm3d(out_channels))
            in_channels = out_channels # Update n of channels for the next layer

        self.flat = nn.Flatten()
        fc_input_size = out_channels * (self._nx // (1<<self.nlayers)) * (self._ny // (1<<self.nlayers)) * (self._nz // (1<<self.nlayers)) #Compute FullyConnected layer input size
        self.fc1 = nn.Linear(fc_input_size, self.nlinear)
        if self._isvae:
            self.mu = nn.Linear(self.nlinear, self._lat_dim)
            self.logvar = nn.Linear(self.nlinear, self._lat_dim)
        else:
            self.z = nn.Linear(self.nlinear, self._lat_dim)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        out = x
        for ilayer, conv_layer in enumerate(self.conv_layers):
            out = conv_layer(out)
            if self.batch_norm:
                out=self.norm_layers[ilayer](out)
            out = self.funcs[ilayer](out)
        out = self.funcs[ilayer+1](self.flat(out))
        out = self.funcs[ilayer+2](self.fc1(out))
        if self._isvae:
            return self.mu(out), self.logvar(out)
        else:
            return self.z(out)


class Decoder3D(nn.Module):
    r"""
    Dencoder3D class for the 3D Convolutional Autoencoder.

    Args:
        nlayers (int): Number of layers in the encoder.
        latent_dim (int): Latent dimension of the encoder.
        nx (int): Height of the input mesh/image.
        ny (int): Width of the input mesh/image.
        nz (int): Depth of the input mesh/image.
        input_channels (int): Number of input channels.
        filter_channels (int): Number of filter channels.
        kernel_size (int): Kernel size for the convolutional layers.
        padding (int): Padding for the convolutional layers.
        activation_funcs (list): List of activation functions.
        nlinear (int): Number of neurons in the linear layer.
        batch_norm (bool): Whether to use batch normalization. Default is ``False``.
        stride (int): Stride for the convolutional layers. Default is ``2``.
        dropout (float): Dropout probability. Default is ``0``.
    """
    def __init__(
        self,
        nlayers: int,
        latent_dim: int,
        nx: int,
        ny: int,
        nz: int,
        input_channels: int,
        filter_channels: int,
        kernel_size: int,
        padding: int,
        activation_funcs: list,
        nlinear: int,
        batch_norm: bool = False,
        stride: int = 2,
        dropout: float = 0,
    ):
        
        super(Decoder3D, self).__init__()       
        
        self.nlayers = nlayers
        self.filt_chan = filter_channels
        self.in_chan = input_channels
        self.lat_dim = latent_dim
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.funcs = activation_funcs
        self.nlinear = nlinear
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(in_features=self.lat_dim, out_features=self.nlinear)
        fc_output_size = int((self.filt_chan * (1 << (self.nlayers-1)) * self.nx // (1 << self.nlayers) * self.ny // (1 << self.nlayers) * self.nz // (1 << self.nlayers)))
        self.fc2 = nn.Linear(in_features=self.nlinear, out_features=fc_output_size)

        # List to hold the transposed convolutional layers
        self.deconv_layers = nn.ModuleList()
        self.norm_layers   = nn.ModuleList()
        in_channels = self.filt_chan * (1 << self.nlayers-1)
        for i in range(self.nlayers-1, 0, -1):
            out_channels = self.filt_chan * (1 << (i - 1))  # Compute output channels
            deconv_layer = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
            self.deconv_layers.append(deconv_layer)
            if self.batch_norm:
                self.norm_layers.append(nn.BatchNorm3d(in_channels))
            in_channels = out_channels  # Update in_channels for the next layer
        out_channels = self.in_chan
        deconv_layer = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        self.deconv_layers.append(deconv_layer)
        if self.batch_norm:
            self.norm_layers.append(nn.BatchNorm3d(in_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.funcs[self.nlayers+1](self.fc1(x))
        out = self.funcs[self.nlayers](self.fc2(out))
        out = out.view(out.size(0), self.filt_chan * (1 << (self.nlayers-1)), int(self.nx // (1 << self.nlayers)), int(self.ny // (1 << self.nlayers)), int(self.nz // (1 << self.nlayers)))
        for ilayer, (deconv_layer) in enumerate(self.deconv_layers[:-1]):
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[self.nlayers-ilayer-1](deconv_layer(out))
        return self.deconv_layers[-1](out)
