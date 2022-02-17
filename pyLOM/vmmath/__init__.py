#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Math operations Module
#
# Last rev: 27/10/2021

__VERSION__ = '1.0.0'

# Vector matrix routines
from .wrapper import transpose, vector_norm, matmul, vecmat, eigen, RMSE, polar, diag, vandermonde
# Averaging routines
from .wrapper import temporal_mean, subtract_mean
# SVD routines
from .wrapper import svd, tsqr_svd
# FFT routines
from .wrapper import fft


del wrapper
