#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# DMD Module
#
# Last rev: 30/09/2021

__VERSION__ = '1.0.0'

# Functions coming from POD
from ..POD.wrapper import temporal_mean, subtract_mean, truncate, residual, RMSE
#from .wrapper import run, truncate, PSD, reconstruct
#from .wrapper import temporal_mean, subtract_mean, svd, residual, power_spectral_density, RMSE

# Functions coming from DMD
from .wrapper import svd, eigen, matrix_split

del wrapper
