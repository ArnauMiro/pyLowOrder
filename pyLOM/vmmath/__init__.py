#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Math operations Module
#
# Last rev: 27/10/2021

__VERSION__ = '1.5.0'

# Vector matrix routines
from .wrapper import transpose, vector_norm, matmul, matmulp, vecmat, argsort, eigen, polar, cholesky, vandermonde, conj, diag, inv, flip, vandermondeTime
# Averaging routines
from .wrapper import temporal_mean, subtract_mean, RMSE
# SVD routines
from .wrapper import qr, svd, tsqr, tsqr_svd
# FFT routines
from .wrapper import fft
# Cell center routines
from .wrapper import cellCenters


del wrapper
