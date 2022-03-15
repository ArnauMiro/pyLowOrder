#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# DMD Module
#
# Last rev: 30/09/2021

__VERSION__ = '1.0.0'

# Functions coming from POD
#from ..POD.wrapper import temporal_mean, subtract_mean, truncate, residual, RMSE
#from .wrapper import run, truncate, PSD, reconstruct
#from .wrapper import temporal_mean, subtract_mean, svd, residual, power_spectral_density, RMSE

# Functions coming from DMD
from .wrapper import run#frequency_damping, mode_computation, amplitude_jovanovic, order_modes, reconstruction_jovanovic
from .plots import ritzSpectrum, amplitudeFrequency, dampingFrequency, plotMode, plotResidual, animateFlow
#from .wrapper import svd, eigen, matrix_split, project_POD_basis, build_complex_eigenvectors
#from .wrapper import frequency_damping, mode_computation, amplitude_jovanovic, order_modes, vandermonde

#del wrapper
