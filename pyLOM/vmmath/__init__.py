#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Math operations Module
#
# Last rev: 27/10/2021

# Vector matrix routines
from .maths      import transpose, vector_sum, vector_norm, vector_mean, matmul, matmulp, vecmat, argsort, eigen, polar, cholesky, vandermonde, conj, diag, inv, flip, vandermondeTime
# Averaging routines
from .averaging  import temporal_mean, subtract_mean
# Truncation routines
from .truncation import compute_truncation_residual, energy
# Statistics routines
from .stats      import RMSE, MAE, r2
# SVD routines
from .svd        import qr, svd, tsqr, randomized_qr, init_qr_streaming, update_qr_streaming, tsqr_svd, randomized_svd
# FFT routines
from .fft        import hammwin, fft
# Geometry and mesh routines
from .geometric  import cellCenters, normals, euclidean_d
# Regression routines
from .regression import least_squares, ridge_regresion


del maths, averaging, truncation, stats, geometric, regression
