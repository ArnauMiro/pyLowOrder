#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for DMD.
#
# Last rev: 27/03/2026
from __future__ import print_function

import numpy as np

from ..utils.gpu import cp
from ..vmmath    import vecmat, matmul, temporal_mean, subtract_mean, svd, tsqr_svd, transpose, eigen, cholesky, diag, polar, vandermonde, conj, inv, flip, matmulp, vandermondeTime, vector_norm
from ..utils     import cr_nvtx as cr, cr_start, cr_stop

@cr('RES.run')
def run(Phi, muReal, muImag, bJov, dt, delta, freq, f, Q=None):
    p = cp if type(Phi) is cp.ndarray else np

    # Normalization of modes (?)
    # Matrix = transpose(vecmat(bJov, transpose(Phi)))
    # for ii in range(len(Matrix[0,:])):
    #     Matrix[:,ii] = Matrix[:,ii] / vector_norm(Phi[:,ii]) 

    Omega = delta + 1j * freq
    H = 1 / (-1j * f - Omega) 

    if Q is None: 

        # Proper treatment (?)
        # Qhat = matmulp(transpose(conj(Phi)), Phi) 
        # Fhat = cholesky(Qhat)
        # Fhat_inv = inv(Fhat)
        # Hhat = matmul(Fhat, vecmat(H, Fhat_inv))
        # U, S, VT = svd(Hhat)
        
        U, S, VT = svd(diag(H))
        V = transpose(conj(VT))

        # U_res = matmul(matmul(Phi, Fhat_inv),U)
        # V_res = matmul(matmul(Phi, Fhat_inv),V)

        U_res = matmul(Phi,U)
        V_res = matmul(Phi,V)


    else:

        Qhat = matmulp(transpose(conj(Phi)), vecmat(Q, Phi))
        Fhat = cholesky(Qhat)
        Fhat_inv = inv(Fhat)

        Hhat = matmul(Fhat, vecmat(H, Fhat_inv))
        U, S, VT = svd(Hhat)
        V = transpose(conj(VT))

        U_res = matmul(matmul(Phi, Fhat_inv),U)
        V_res = matmul(matmul(Phi, Fhat_inv),V)

    return U_res, S, V_res


@cr('RES.weighting')
def weighting(m, v_dims = 1, gamma=1.4, cp=1004.0, T=1, rho=None, compressible=False): # Not well parallelized
    p = cp if type(rho) is cp.ndarray else np

    R = cp * (1 - gamma)
    c = p.sqrt(gamma * R * T) # How do I find T?

    V = np.array([])
    # Calcular els dV
    dV = np.ones(len(m.xyz[:,0])) # How do I calculate the volumes?

    # Calcular Q
    if compressible == False:
        for ii in range(v_dims):
            V = np.append(V, dV)
        Q = 0.5 * rho * V

    if compressible == True:
        for ii in range(v_dims):
            V = np.append(V, rho * dV)
        V = np.append(V, c / (gamma * rho) * dV)
        Q = 0.5 * V