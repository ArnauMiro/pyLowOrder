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
from ..vmmath    import vecmat, matmul, svd, cholesky, inv, matmulp, dagger, concatenate, linear_operator, separate, resolvent
from ..utils     import cr_nvtx as cr, cr_start, cr_stop

@cr('RES.run')
def run(Phi, delta, omega, f, W=None):
    '''
    Resolvent Analysis of snapshot matrix X
    Inputs:
        - X[ndims*nmesh,n_temp_snapshots]: data matrix
        - delta: damping ratio of each mode
        - omega: frequency of each mode
        - f: target frequency
        - W: diagonal matrix (1D) of the inner-product
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
    if W is None: 
        What = matmulp(dagger(Phi), Phi) 
    
    else:
        What = matmulp(dagger(Phi), vecmat(W, Phi))
    
    Fhat = dagger(cholesky(What))
    Fhat_inv = inv(Fhat)

    # Solve the optimization problem
    Hhat = matmul(Fhat, vecmat(H, Fhat_inv))
    U, S, VT = svd(Hhat)
    V = dagger(VT)

    # Project the solution
    U_res = matmul(Phi, matmul(Fhat_inv, U))
    V_res = matmul(Phi, matmul(Fhat_inv, V))

    return U_res, S, V_res

def run_new(X, w, r, remove_mean=True, W=None):

    # Prepare matrices and calculate the linear operator
    X_c = X.copy()
    W_sqrt = np.sqrt(W) if W is not None else None
    if (type(X) is list):
        if W is not None:
            for M, ii in enumerate(X):
                X_c[ii] = vecmat(W_sqrt, M)
        Y, Z = concatenate(X_c, remove_mean=remove_mean)
        U1, S1, VT1, Atilde = linear_operator(Y, Z, r)

    else:
        if W is not None:
            X_c = vecmat(W_sqrt, X)
        Y, Z = separate(X_c, remove_mean=remove_mean)
        U1, S1, VT1, Atilde = linear_operator(Y, Z, r)

    del S1, VT1

    # Calculate the resolvent of the linear operator
    U2, S, V2 = resolvent(Atilde, w)

    # Project the solution
    U = matmul(U1, U2)
    V = matmul(U1, V2)

    if W is not None:
        U = vecmat(1/W_sqrt, U)
        V = vecmat(1/W_sqrt, V)

    return U, S, V
