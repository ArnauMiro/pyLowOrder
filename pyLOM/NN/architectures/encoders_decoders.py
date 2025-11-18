#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Encoder-Decoder architecture for NN Module
#
# Last rev: 09/10/2024

import torch.nn as nn
import torch

class Encoder1D(nn.Module):
    r"""
    Encoder1D class for the 1D Convolutional Autoencoder.

    Args:
        nlayers (int): Number of layers in the encoder.
        latent_dim (int): Latent dimension of the encoder.
        nh (int): Height of the input.
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

    def __init__(self, nlayers, latent_dim, input_length, input_channels,
                 filter_channels, kernel_size, padding, activation_funcs,
                 nlinear, batch_norm=True, stride=2, dropout=0, vae=True):
        super(Encoder1D, self).__init__()

        self.nlayers    = nlayers
        self.filt_chan  = filter_channels
        self.in_chan    = input_channels
        self.lat_dim    = latent_dim
        self.input_len  = input_length
        self.isvae      = vae
        self.funcs      = activation_funcs
        self.nlinear    = nlinear
        self.batch_norm = batch_norm
        self.dropout    = nn.Dropout(p=dropout)

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        in_channels = self.in_chan

        for ilayer in range(self.nlayers):
            out_channels = self.filt_chan * (1 << ilayer)
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv_layers.append(conv_layer)
            if self.batch_norm:
                self.norm_layers.append(nn.GroupNorm(out_channels, out_channels))
            in_channels = out_channels

        # Compute the length after all downsampling
        conv_length = self.input_len
        for _ in range(self.nlayers):
            conv_length = (conv_length + 2 * padding - kernel_size) // stride + 1

        self.flat     = nn.Flatten()
        fc_input_size = out_channels * conv_length
        self.fc1      = nn.Linear(fc_input_size, self.nlinear)
        
        if self.isvae:
            self.mu     = nn.Linear(self.nlinear, self.lat_dim)
            self.logvar = nn.Linear(self.nlinear, self.lat_dim)
        else:
            self.z = nn.Linear(self.nlinear, self.lat_dim)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = x
        for ilayer, conv_layer in enumerate(self.conv_layers):
            out = conv_layer(out)
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[ilayer](out)
        out = self.funcs[ilayer + 1](self.flat(out))
        out = self.funcs[ilayer + 2](self.fc1(out))
        if self.isvae:
            return self.mu(out), self.logvar(out)
        else:
            return self.z(out)

class Decoder1D(nn.Module):
    r"""
    Decoder1D class for the 1D Convolutional Autoencoder.

    Args:
        nlayers (int): Number of layers in the encoder.
        latent_dim (int): Latent dimension of the encoder.
        nh (int): Height of the input mesh/image.
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
    def __init__(self, nlayers, latent_dim, input_length, input_channels,
                 filter_channels, kernel_size, padding, activation_funcs,
                 nlinear, batch_norm=True, stride=2, dropout=0):
        super(Decoder1D, self).__init__()

        self.nlayers    = nlayers
        self.filt_chan  = filter_channels
        self.in_chan    = input_channels
        self.lat_dim    = latent_dim
        self.input_len  = input_length
        self.funcs      = activation_funcs
        self.nlinear    = nlinear
        self.batch_norm = batch_norm
        self.dropout    = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(self.lat_dim, self.nlinear)

        # Compute the length after deconvolutions (in reverse)
        deconv_length = self.input_len
        for _ in range(self.nlayers):
            deconv_length = (deconv_length - 1) // stride + 1

        conv1d_channels = self.filt_chan * (1 << (self.nlayers - 1))
        self.fc2 = nn.Linear(self.nlinear, conv1d_channels * deconv_length)

        self.deconv_layers = nn.ModuleList()
        self.norm_layers   = nn.ModuleList()
        in_channels = conv1d_channels

        for i in range(self.nlayers - 1, 0, -1):
            out_channels = self.filt_chan * (1 << (i - 1))
            deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
            self.deconv_layers.append(deconv)
            if self.batch_norm:
                self.norm_layers.append(nn.GroupNorm(in_channels,in_channels))
            in_channels = out_channels

        # Final layer: back to input channels
        deconv_final = nn.ConvTranspose1d(in_channels, self.in_chan, kernel_size, stride, padding)
        self.deconv_layers.append(deconv_final)
        if self.batch_norm:
            self.norm_layers.append(nn.GroupNorm(in_channels,in_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.funcs[self.nlayers + 1](self.fc1(x))
        out = self.funcs[self.nlayers](self.fc2(out))

        conv1d_channels = self.filt_chan * (1 << (self.nlayers - 1))
        deconv_length = out.shape[1] // conv1d_channels
        out = out.view(out.size(0), conv1d_channels, deconv_length)

        for ilayer, deconv_layer in enumerate(self.deconv_layers[:-1]):
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[self.nlayers - ilayer - 1](deconv_layer(out))

        return self.deconv_layers[-1](out)


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

    def forward(self, x:torch.Tensor): 
        r'''
		Do a forward evaluation of the data.

		Args:
			x (torch.Tensor): input data to the neural network.

		Returns:
		    (torch.Tensor): Prediction of the neural network.
		''' 
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


class FullyConnectedEncoder2D(nn.Module):
    r"""
    FullyConnectedEncoder2D class for the 2D fully-connected Autoencoder.

    Args:
        hidden_layer_sizes (list): Layer sizes in the encoder.
        latent_dim (int): Latent dimension of the encoder.
        in_size (int): input size.
        activation_funcs (list): List of activation functions.
        vae (bool): Wheather the encoder is going to be used on a VAE or not. Default is ``False``.
    """
    def __init__(
        self,
        hidden_layer_sizes: list,
        lat_dim: int,
        in_size: int,
        activation_funcs: list,
        vae: bool = False,
    ):
        super(FullyConnectedEncoder2D, self).__init__()

        self.hidden_layer_sizes    = hidden_layer_sizes
        self.lat_dim               = lat_dim
        self.in_size               = in_size
        self.isvae                 = vae
        self.funcs                 = activation_funcs
        
        self.encoding_layers = nn.ModuleList()

        if len(self.hidden_layer_sizes) != len(self.funcs):
            raise ValueError("Incorrect number of layers! 'hidden_layer_sizes' and 'funcs' must have the same length.")
        
        current_dim = self.in_size
        for idx, hidden_dim in enumerate(self.hidden_layer_sizes):
            self.encoding_layers.append(nn.Linear(current_dim, hidden_dim))
            self.encoding_layers.append(self.funcs[idx])
            current_dim = hidden_dim

        if self.isvae:
            self.encoding_layers.append(nn.Linear(current_dim, self.lat_dim*2))
        else:
            self.encoding_layers.append(nn.Linear(current_dim, self.lat_dim))

        self._reset_parameters()
        return None
    
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
            else:
                pass
        return None

    def forward(self, x:torch.Tensor): 
        r'''
		Do a forward evaluation of the data.

		Args:
			x (torch.Tensor): input data to the neural network.

		Returns:
		    (torch.Tensor): Prediction of the neural network.
		''' 
        out = x
        for ilayer, layer in  enumerate(self.encoding_layers):
            out = layer(out)
        if self.isvae:
            mu, logvar = torch.chunk(input=out, chunks=2, dim=-1)
            return mu, logvar
        else:
            return out


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

    def forward(self, x:torch.Tensor):
        r'''
		Do a forward evaluation of the data.

		Args:
			x (torch.Tensor): input data to the neural network.

		Returns:
			(torch.Tensor): Prediction of the neural network.
		'''
        out = self.funcs[self.nlayers+1](self.fc1(x))
        out = self.funcs[self.nlayers](self.fc2(out))
        out = out.view(out.size(0), self.filt_chan * (1 << (self.nlayers-1)), int(self.nh // (1 << self.nlayers)), int(self.nw // (1 << self.nlayers)))
        for ilayer, (deconv_layer) in enumerate(self.deconv_layers[:-1]):
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[self.nlayers-ilayer-1](deconv_layer(out))
        return self.deconv_layers[-1](out)


class FullyConnectedDecoder2D(nn.Module):
    r"""
    FullyConnectedDecoder2D class for the 2D fully-connected Autoencoder.

    Args:
        hidden_layer_sizes (list): Layer sizes in the encoder.
        latent_dim (int): Latent dimension of the encoder.
        out_size (int): output size.
        activation_funcs (list): List of activation functions.
    """
    def __init__(
        self,
        hidden_layer_sizes: list,
        lat_dim: int,
        out_size: int,
        activation_funcs: list,
    ):
        super(FullyConnectedDecoder2D, self).__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.lat_dim            = lat_dim
        self.out_size           = out_size
        self.funcs              = activation_funcs
        
        self.decoding_layers = nn.ModuleList()

        if len(self.hidden_layer_sizes) != len(self.funcs):
            raise ValueError("Incorrect number of layers! 'hidden_layer_sizes' and 'funcs' must have the same length.")

        current_dim = self.lat_dim
        for idx, hidden_dim in enumerate(self.hidden_layer_sizes):
            self.decoding_layers.append(nn.Linear(current_dim, hidden_dim))
            self.decoding_layers.append(self.funcs[idx])
            current_dim = hidden_dim
        self.decoding_layers.append(nn.Linear(current_dim, out_size))
        self._reset_parameters()
        return None
    
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
            else:
                pass
        return None

    def forward(self, x:torch.Tensor): 
        r'''
		Do a forward evaluation of the data.

		Args:
			x (torch.Tensor): input data to the neural network.

		Returns:
		    (torch.Tensor): Prediction of the neural network.
		''' 
        out = x
        for ilayer, layer in  enumerate(self.decoding_layers):
            out = layer(out)
        return out


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
    
    def forward(self, x:torch.Tensor):
        r'''
		Do a forward evaluation of the data.

		Args:
			x (torch.Tensor): input data to the neural network.

		Returns:
			(torch.Tensor): Prediction of the neural network.
		'''
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

    def forward(self, x:torch.Tensor):
        r'''
		Do a forward evaluation of the data.

		Args:
			x (torch.Tensor): input data to the neural network.

		Returns:
			(torch.Tensor): Prediction of the neural network.
		'''
        out = self.funcs[self.nlayers+1](self.fc1(x))
        out = self.funcs[self.nlayers](self.fc2(out))
        out = out.view(out.size(0), self.filt_chan * (1 << (self.nlayers-1)), int(self.nx // (1 << self.nlayers)), int(self.ny // (1 << self.nlayers)), int(self.nz // (1 << self.nlayers)))
        for ilayer, (deconv_layer) in enumerate(self.deconv_layers[:-1]):
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[self.nlayers-ilayer-1](deconv_layer(out))
        return self.deconv_layers[-1](out)

class ShallowDecoder(nn.Module):
	r"""
    Decoder used for the SHRED architecture. 

    Args:
        output_size (int): Number of POD modes to predict.
        hidden_size (int): Dimension of the LSTM hidden layers.
		decoder_sizes (list): Integer list of the decoder layer sizes.
        dropout (float): Dropout probability for the decoder.
    """
	def __init__(self, output_size:int, hidden_size:int, decoder_sizes:list, dropout:float):
		super(ShallowDecoder, self).__init__()
		decoder_sizes.insert(0, hidden_size)
		decoder_sizes.append(output_size)
		self.layers = nn.ModuleList()

		for i in range(len(decoder_sizes)-1):
			self.layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
			if i != len(decoder_sizes)-2:
				self.layers.append(nn.Dropout(dropout))
				self.layers.append(nn.ReLU())

	def forward(self, output:torch.Tensor):
		r'''
		Do a forward evaluation of the data.

		Args:
			x (torch.Tensor): input data to the neural network.

		Returns:
			(torch.Tensor): Prediction of the neural network.
		'''
		for layer in self.layers:
			output = layer(output)
		return output
    
class Encoder1DNoLatent(nn.Module):
    r"""
    Encoder1D class for the 1D Convolutional Autoencoder without linear layers to compress the latent space, useful for data compression.

    Args:
        nlayers (int): Number of layers in the encoder.
        input_channels (int): Number of input channels.
        filter_channels (int): Number of filter channels.
        kernel_size (int): Kernel size for the convolutional layers.
        padding (int): Padding for the convolutional layers.
        activation_funcs (list): List of activation functions.
        batch_norm (bool): Whether to use batch normalization. Default is ``False``.
        stride (int): Stride for the convolutional layers. Default is ``2``.
        dropout (float): Dropout probability. Default is ``0``.
        vae (bool): Wheather the encoder is going to be used on a VAE or not. Default is ``False``.
    """
    def __init__(self, nlayers, input_length, input_channels,
                 filter_channels, kernel_size, padding, activation_funcs,
                 batch_norm=False, stride=2, dropout=0, vae=False):
        super(Encoder1DNoLatent, self).__init__()

        self.nlayers    = nlayers
        self.filt_chan  = filter_channels
        self.in_chan    = input_channels
        self.input_len  = input_length
        self.isvae      = vae
        self.funcs      = activation_funcs
        self.batch_norm = batch_norm
        self.dropout    = nn.Dropout(p=dropout)

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        in_channels = self.in_chan

        for ilayer in range(self.nlayers):
            out_channels = self.filt_chan * (1 << ilayer)
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv_layers.append(conv_layer)
            if self.batch_norm:
                self.norm_layers.append(nn.GroupNorm(out_channels, out_channels))
            in_channels = out_channels

        # Compute the length after all downsampling
        conv_length = self.input_len
        for _ in range(self.nlayers):
            conv_length = (conv_length + 2 * padding - kernel_size) // stride + 1

        self.flat     = nn.Flatten()
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = x
        for ilayer, conv_layer in enumerate(self.conv_layers):
            #print(out.shape)
            out = conv_layer(out)
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[ilayer](out)
        out = self.funcs[self.nlayers - ilayer - 1](self.flat(out))
        return out

class Decoder1DNoLatent(nn.Module):
    r"""
    Decoder1D class for the 1D Convolutional Autoencoder without linear layers for latent space compression.

    Args:
        nlayers (int): Number of layers in the encoder.
        input_channels (int): Number of input channels.
        filter_channels (int): Number of filter channels.
        kernel_size (int): Kernel size for the convolutional layers.
        padding (int): Padding for the convolutional layers.
        activation_funcs (list): List of activation functions.
        batch_norm (bool): Whether to use batch normalization. Default is ``False``.
        stride (int): Stride for the convolutional layers. Default is ``2``.
        dropout (float): Dropout probability. Default is ``0``.
    """
    def __init__(self, nlayers, input_length, input_channels,
                 filter_channels, kernel_size, padding, activation_funcs,
                batch_norm=False, stride=2, dropout=0):
        super(Decoder1DNoLatent, self).__init__()

        self.nlayers    = nlayers
        self.filt_chan  = filter_channels
        self.in_chan    = input_channels
        self.input_len  = input_length
        self.funcs      = activation_funcs
        self.batch_norm = batch_norm
        self.dropout    = nn.Dropout(p=dropout)


        # Compute the length after deconvolutions (in reverse)
        deconv_length = self.input_len

        for _ in range(self.nlayers):
            deconv_length = (deconv_length - 1) // stride + 1

        conv1d_channels = self.filt_chan * (1 << (self.nlayers - 1))
        #self.fc1 = nn.Linear(128, deconv_length*conv1d_channels)

        self.deconv_layers = nn.ModuleList()
        self.norm_layers   = nn.ModuleList()
        in_channels = conv1d_channels

        for i in range(self.nlayers - 1, 0, -1):
            out_channels = self.filt_chan * (1 << (i - 1))
            deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
            self.deconv_layers.append(deconv)
            if self.batch_norm:
                self.norm_layers.append(nn.GroupNorm(in_channels,in_channels))
            in_channels = out_channels

        # Final layer: back to input channels
        deconv_final = nn.ConvTranspose1d(in_channels, self.in_chan, kernel_size, stride, padding)
        self.deconv_layers.append(deconv_final)
        if self.batch_norm:
            self.norm_layers.append(nn.GroupNorm(in_channels,in_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        conv1d_channels = self.filt_chan * (1 << (self.nlayers - 1))
        #x   = self.funcs[self.nlayers - 1](self.fc1(x))
        deconv_length = x.shape[1] // conv1d_channels
        out = x.view(x.size(0), conv1d_channels, deconv_length)

        for ilayer, deconv_layer in enumerate(self.deconv_layers[:-1]):
            if self.batch_norm:
                out = self.norm_layers[ilayer](out)
            out = self.funcs[self.nlayers - ilayer - 1](deconv_layer(out))

        return self.deconv_layers[-1](out)