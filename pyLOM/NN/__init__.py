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
from ..utils import MPI_RANK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(int(MPI_RANK % torch.cuda.device_count())) ## Set the correct device number according to MPI_RANK, if not it is only the cuda:0 working

## Flags to enchance performance when using torch compiled with cuda in the backend
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()
ALLOW_TF32=True
PIN_MEMORY = True if torch.cuda.is_available() else False

from ..utils.plots  import plotSnapshot, plotModalErrorBars, plotTimeSeries

from .pipeline      import Pipeline, ClusteredPipeline
from .utils         import Dataset, MinMaxScaler, select_device, betaLinearScheduler, create_results_folder, set_seed

from .optimizer     import OptunaOptimizer

from .stats         import RegressionEvaluator, ClassificationEvaluator
from .callbacks     import EarlyStopper

from .interpolator import Interpolator
from .aerodynamics import global_coeff, jacobians_pressure

from .architectures.mlp               import MLP
from .architectures.kan               import KAN, KAN_SIN, ChebyshevLayer, JacobiLayer, SineLayer
from .architectures.autoencoders      import Autoencoder, FullyConnectedAutoencoder, VariationalAutoencoder, FullyConnectedVariationalAutoencoder
from .architectures.encoders_decoders import Encoder1D, Decoder1D, Encoder2D, Decoder2D, Encoder3D, Decoder3D, ShallowDecoder, Encoder1DNoLatent, Decoder1DNoLatent
from .architectures.pinn              import PINN, BurgersPINN, Euler2DPINN, NavierStokesIncompressible2DPINN, BoundaryCondition
from .architectures.shred             import SHRED
from .architectures.binary_classifier import BinaryClassifier


# Wrapper of the activation functions
def tanh():      return nn.Tanh()
def relu():      return nn.ReLU()
def elu():       return nn.ELU()
def sigmoid():   return nn.Sigmoid()
def leakyRelu(): return nn.LeakyReLU()
def silu():      return nn.SiLU()

from .                            import GAVI

del os, torch