#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for DMD.
#
# Last rev: 20/02/2026
from __future__ import print_function

import numpy as np

from ..utils.gpu import cp
from ..vmmath    import vecmat, matmul, temporal_mean, subtract_mean, svd, tsqr_svd, transpose, eigen, cholesky, diag, polar, vandermonde, conj, inv, flip, matmulp, vandermondeTime, vector_norm
from ..utils     import cr_nvtx as cr, cr_start, cr_stop

@cr('RES.run')
def run(Phi, muReal, muImag, bJov, dt, delta, freq, f, Q=None):
    p = cp if type(Phi) is cp.ndarray else np

    # Treatment of the parameters needed for future matrices
    mu = muReal + 1j * muImag
    Amplitude = bJov
    # Frequency = np.angle(mu) / (dt * param)
    # GrowthRate = np.log(np.abs(mu)) / dt
    # Omega = GrowthRate + 1j * Frequency
    Frequency = freq
    GrowthRate = delta
    Omega = GrowthRate + 1j * Frequency
    # Normalization of modes (?)
    Matrix = transpose(vecmat(bJov, transpose(Phi)))
    for ii in range(len(Matrix[0,:])):
        Matrix[:,ii] = Matrix[:,ii] / vector_norm(Phi[:,ii]) 
    # Calculating Cholesky decomposition for Q
    I = p.eye(len(mu))
    Lambda = diag(Omega)

    if Q == None:
        H = inv(-1j * f * I - Lambda)
        U, S, VT = svd(H)
        V = transpose(conj(VT))

        U_res = matmul(Phi,U)
        V_res = matmul(Phi,V)

    else:
        Qhat = matmul(matmul(transpose(conj(Matrix)), Q), Matrix)
        # Qhat = matmul(transpose(conj(Matrix)), Matrix)
        Fhat = cholesky(Qhat)

        # Calculating modes for desired frequency
        # H = matmul(matmul(Fhat, inv(-1j * f * I - Lambda)), np.linalg.pinv(Fhat))
        H = matmul(matmul(Fhat, inv(-1j * f * I - Lambda)), inv(Fhat))
        U, S, VT = svd(H)
        V = transpose(conj(VT))

        # URes = matmul(matmul(Matrix, np.linalg.pinv(Fhat)),U)
        # VRes = matmul(matmul(Matrix, np.linalg.pinv(Fhat)),V)

        U_res = matmul(matmul(Matrix, inv(Fhat)),U)
        V_res = matmul(matmul(Matrix, inv(Fhat)),V)

    # U_abs = np.abs(URes)
    # V_abs = np.abs(VRes)
    # U_norm = np.zeros_like(U_abs)
    # V_norm = np.zeros_like(V_abs)
    # for jj in range(len(U_abs[0])):
    #     U_norm[:,jj] = (U_abs[:,jj] - np.min(U_abs[:,jj])) / (np.max(U_abs[:,jj]) - np.min(U_abs[:,jj]))
    #     V_norm[:,jj] = (V_abs[:,jj] - np.min(V_abs[:,jj])) / (np.max(V_abs[:,jj]) - np.min(V_abs[:,jj]))

    return U_res, S, V_res

@cr('RES.run_mine')
def run_mine(Phi, muReal, muImag, delta, frequency, dt, f):
    p = cp if type(Phi) is cp.ndarray else np

    # Treatment of the parameters needed for future matrices
    GrowthRate, Frequency = delta, frequency
    # Frequency = Frequency / (2 * p.pi)
    Omega = GrowthRate + 1j * Frequency

    # Normalization of the modes
    # for ii in range(len(Phi[0,:])):
    #     Phi[:,ii] = Phi[:,ii] / vector_norm(Phi[:,ii]) 
    I = p.eye(len(Omega))
    Lambda = diag(Omega)

    # Calculating modes for desired frequency
    H = inv(-1j * f * I - Lambda)
    U, S, VT = svd(H)
    V = transpose(conj(VT))

    U_res = matmul(Phi,U)
    V_res = matmul(Phi,V)

    # U_abs = p.abs(U_res)
    # V_abs = p.abs(V_res)
    # U_norm = np.zeros_like(U_abs)
    # V_norm = np.zeros_like(V_abs)
    # for jj in range(len(U_abs[0])):
    #     U_norm[:,jj] = (U_abs[:,jj] - np.min(U_abs[:,jj])) / (np.max(U_abs[:,jj]) - np.min(U_abs[:,jj]))
    #     V_norm[:,jj] = (V_abs[:,jj] - np.min(V_abs[:,jj])) / (np.max(V_abs[:,jj]) - np.min(V_abs[:,jj]))

    return U_res, S, V_res