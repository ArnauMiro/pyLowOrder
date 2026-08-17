#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Python interface for DMD.
#
# Last rev: 30/09/2021
from __future__ import print_function

import numpy as np

from ..utils.gpu import cp
from ..vmmath    import vecmat, matmul, temporal_mean, subtract_mean, tsqr_svd, transpose, eigen, cholesky, diag, polar, vandermonde, conj, inv, flip, matmulp, vandermondeTime, svd
from ..POD       import truncate
from ..utils     import cr_nvtx as cr, cr_start, cr_stop, worksplit, MPI_RANK, mpi_gather, raiseError
from ..          import pprint

def _order_modes(muReal, muImag, Phi, bJov):
    '''
    Order the modes according to its amplitude, forcing that in case of a conjugate eigenvalue, the positive part always is the first one
    '''
    cnp    = cp if type(muReal) is cp.ndarray else np
    muReal = muReal[flip(cnp.abs(bJov).argsort())]
    muImag = muImag[flip(cnp.abs(bJov).argsort())]
    Phi    = transpose(transpose(Phi)[flip(cnp.abs(bJov).argsort())])
    bJov   = bJov[flip(cnp.abs(bJov).argsort())]
    p = False
    for ii in range(muImag.shape[0]):
        if p == True:
            p = False
            continue
        iimag = muImag[ii]
        if iimag < 0:
            muImag[ii]        =  muImag[ii+1]
            muImag[ii+1]      = -muImag[ii]
            bJov.imag[ii]     =  bJov.imag[ii+1]
            bJov.imag[ii+1]   = -bJov.imag[ii]
            Phi.imag[:,ii]    =  Phi.imag[:,ii+1]
            Phi.imag[:,ii+1]  = -Phi.imag[:,ii+1]
            p = True
            continue
        if iimag > 0:
            p = True
            continue
    return muReal, muImag, Phi, bJov


@cr('DMD.run')
def run(X=None, Y=None, Z=None, tol=1e-6, remove_mean=False):
    '''
    DMD analysis of snapshot matrix X
    Inputs:
        - X[ndims*nmesh,n_temp_snapshots]: data matrix
        - remove_mean:                     whether or not to remove the mean flow

    Returns:
        - Phi:      DMD Modes
        - muReal:   Real part of the eigenvalues
        - muImag:   Imaginary part of the eigenvalues
        - b:        Amplitude of the DMD modes
        - X_DMD:    Reconstructed flow
    '''
    if X is not None:
        if Y is not None or Z is not None:
            raiseError("In DMD.run either X or (Y and Z) must be provided")

        if (type(X) is list):
            if len(X) == 0:
                raiseError("In DMD.run the argument cannot be an empty list")

            n = X[0].shape[0]

            if not all([M.shape[0] == n for M in X]):
                raiseError("In DMD.run all matrices of list must have same number"
                    " of rows")

            m = sum([M.shape[1] - 1 for M in X])

            Y = np.zeros((n, m), dtype=X[0].dtype)
            Z = np.zeros((n, m), dtype=X[0].dtype)

            offset = 0
            for M in X:
                if remove_mean:
                    cr_start('DMD.temporal_mean',0)
                    mean = temporal_mean(M)

                    Y[:,offset:offset + M.shape[1] - 1] = \
                        subtract_mean(M[:,:-1], mean)
                    Z[:,offset:offset + M.shape[1] - 1] = \
                        subtract_mean(M[:,1:], mean)
                    offset += M.shape[1] - 1
                    cr_stop('DMD.temporal_mean',0)
                else:
                    Y[:,offset:offset + M.shape[1] - 1] = M[:,:-1]
                    Z[:,offset:offset + M.shape[1] - 1] = M[:,1:]
        else:
            # Remove temporal mean or not, depending on the user choice
            if remove_mean:
                cr_start('DMD.temporal_mean',0)
                #Compute temporal mean
                X_mean = temporal_mean(X)
                #Subtract temporal mean
                Y = subtract_mean(X, X_mean)
                Z = Y[:,1:]
                Y = Y[:,:-1]
                cr_stop('DMD.temporal_mean',0)
            else:
                Y = X[:,:-1]
                Z = X[:,1:]

    else:
        if Y is None or Z is None:
            raiseError("In DMD.run either X or (Y and Z) must be provided")

        if remove_mean:
            raiseError("In DMD.run if Y and Z are provided remove_mean must be False")

        if Y.shape != Z.shape:
            raiseError("In DMD.run if Y and Z are provided, must be of the same shape")

    # Compute SVD
    cr_start('DMD.SVD',0)
    pprint(0, "In DMD.run: Computing SVD...", flush=True)
    U, S, VT = tsqr_svd(Y)
    pprint(0, "In DMD.run: Computed SVD", flush=True)
    cr_stop('DMD.SVD',0)
    nmodes = 1
    while nmodes < S.shape[0]:
        if S[nmodes] / S[0] < tol:
            break
        nmodes += 1

    # Truncate according to residual
    cr_start('DMD.truncate', 0)
    U, S, VT = truncate(U, S, VT, nmodes)
    cr_stop('DMD.truncate', 0)

    # Project A (Jacobian of the snapshots) into POD basis
    cr_start('DMD.linear_mapping',0)
    aux1   = matmulp(transpose(U), Z)
    aux2   = transpose(vecmat(1./S, VT))
    Atilde = matmul(aux1, aux2)
    cr_stop('DMD.linear_mapping',0)

    return U, S, VT, Atilde

@cr('DMD.resolvent_analysis')
def resolvent_analysis(A:np.ndarray, w:np.ndarray, n_modes:int=1):
    if np.shape(A)[0] != np.shape(A)[1]:
        pyLOM.utils.raiseError("In resolvent_analysis, the matrix is not square")

    n = np.shape(A)[0]
    start, end = worksplit(0, len(w), MPI_RANK)
    m = end - start
    amplitudes = np.zeros((m * n_modes,))
    direct_modes = np.zeros((n,  m * n_modes), dtype=np.complex128)
    adjoint_modes = np.zeros((n, m * n_modes), dtype=np.complex128)

    complexA = np.array(A, dtype=np.complex128)
    I = np.identity(n)

    for i in range(m):
        shifted = complexA - I * w[i + start]
        U, S, VT = svd(shifted)

        for j in range(n_modes):
            amplitudes[i*n_modes + j] = 1 / S[n - 1 - j]
            direct_modes[:, i*n_modes + j]  = VT[n - 1 - j, :].reshape((-1,))
            adjoint_modes[:, i*n_modes + j] = U [:, n - 1 - j].reshape((-1,))

    amplitudes = mpi_gather(amplitudes, all=True)
    direct_modes = mpi_gather(direct_modes.T, all=True)
    adjoint_modes = mpi_gather(adjoint_modes.T, all=True)

    return amplitudes, direct_modes.T, adjoint_modes.T

@cr('DMD.frequency_damping')
def frequency_damping(real, imag, dt):
    '''
    Computation of the damping ratio and the frequency of each mode
    '''
    p = cp if type(real) is cp.ndarray else np
    mod, arg = polar(real, imag) #Create vmmath/complex.c?
    # Computation of the damping ratio of the mode
    delta = p.log(mod)/dt
    # Computation of the frequency of the mode
    omega = arg/dt
    return delta, omega

@cr('DMD.mode_computation')
def mode_computation(X, V, S, W):
    '''
    Computation of DMD Modes
    '''
    p = cp if type(X) is cp.ndarray else np
    return  matmul(matmul(matmul(X, transpose(V)), diag(1/S)), p.abs(W))

@cr('DMD.reconstruction_jovanovic')
def reconstruction_jovanovic(Phi, real, imag, t, bJov):
    '''
    Reconstruction of the DMD modes according to the Jovanovic method
    '''
    Vand = vandermondeTime(real, imag, real.shape[0], t)
    return matmul(Phi, matmul(diag(bJov), Vand)).real
