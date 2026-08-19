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
from ..vmmath    import vecmat, matmul, svd, cholesky, inv, matmulp, dagger
from ..utils     import cr_nvtx as cr, cr_start, cr_stop

@cr('RES.run')
def run(Phi, delta, omega, f, Q=None):
    '''
    Resolvent Analysis of snapshot matrix X
    Inputs:
        - X[ndims*nmesh,n_temp_snapshots]: data matrix
        - delta: damping ratio of each mode
        - omega: frequency of each mode
        - f: target frequency
        - Q: weighting matrix
    Returns:
        - U_res: response modes
        - S: emergy gains
        - V_res: forcing modes
    '''
    p = cp if type(Phi) is cp.ndarray else np

    # Compute the resolvent operator
    Omega = delta + 1j * omega
    H = 1 / (-1j * f - Omega) 
    
    # Compute the metric
    if Q is None: 
        Qhat = matmulp(dagger(Phi), Phi) 
    
    else:
        Qhat = matmulp(dagger(Phi), vecmat(Q, Phi))
    
    Fhat = dagger(cholesky(Qhat))
    Fhat_inv = inv(Fhat)

    # Solve the optimization problem
    Hhat = matmul(Fhat, vecmat(H, Fhat_inv))
    U, S, VT = svd(Hhat)
    V = dagger(VT)

    # Project the solution
    U_res = matmul(Phi, matmul(Fhat_inv, U))
    V_res = matmul(Phi, matmul(Fhat_inv, V))

    return U_res, S, V_res