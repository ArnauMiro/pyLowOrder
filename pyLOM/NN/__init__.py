#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN Module
#
# Last rev: 09/10/2024

# Supress prints from tensorflow
import os, torch, torch.nn as nn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from ..utils.plots  import plotSnapshot

from .pipeline      import Pipeline
from .utils         import Dataset, MinMaxScaler, select_device, betaLinearScheduler, create_results_folder, set_seed

from .optimizer     import OptunaOptimizer

from .stats         import RegressionEvaluator
from .callbacks     import EarlyStopper

from .architectures.mlp               import MLP
from .architectures.kan               import KAN, ChebyshevLayer, JacobiLayer
from .architectures.autoencoders      import Autoencoder, VariationalAutoencoder
from .architectures.encoders_decoders import Encoder2D, Decoder2D, Encoder3D, Decoder3D
from .architectures.pinn              import PINN, BurgersPINN, Euler2DPINN, NavierStokesIncompressible2DPINN, BoundaryCondition



# Wrapper of the activation functions
def tanh():      return nn.Tanh()
def relu():      return nn.ReLU()
def elu():       return nn.ELU()
def sigmoid():   return nn.Sigmoid()
def leakyRelu(): return nn.LeakyReLU()
def silu():      return nn.SiLU()

del os, torch
