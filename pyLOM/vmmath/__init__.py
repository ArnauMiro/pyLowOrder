#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Math operations Module
#
# Last rev: 27/10/2021

# Vector matrix routines
from .maths          import transpose, vector_sum, vector_norm, vector_mean, matmul, matmulp, vecmat, argsort, eigen, polar, cholesky, vandermonde, conj, diag, inv, flip, vandermondeTime
# Averaging routines
from .averaging      import temporal_mean, subtract_mean, temporal_variance, norm_variance
# Truncation routines
from .truncation     import compute_truncation_residual, energy
# Statistics routines
from .stats          import RMSE, MAE, r2, MRE_array
# SVD routines
from .svd            import qr, svd, tsqr, randomized_qr, randomized_qr2, init_qr_streaming, update_qr_streaming, tsqr_svd, randomized_svd
# FFT routines
from .fft            import hammwin, fft
# Geometry and mesh routines
from .geometric      import cellCenters, normals, euclidean_d, wall_normals, edge_to_cells, cell_adjacency, fix_normals_coherence
# Regression routines
from .regression     import least_squares, ridge_regresion
# Data processing module
from .dataprocessing import data_splitting, time_delay_embedding, find_random_sensors


del maths, averaging, truncation, stats, geometric, regression
