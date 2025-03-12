#!/usr/bin/env python
#
# Manifold learning methodologies.
#
# Last revision: 11/02/2025
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigh

from ..utils.gpu import cp
from ..vmmath    import euclidean_d
from ..utils     import cr_nvtx as cr, raiseError, pprint


@cr('MANIFOLD.isomap')
def isomap(X:np.ndarray, dims:int, n_size:int, comp:int = 1 ,verbose:bool = True):
    """
    Computes Isomap embedding using the algorithm of Tenenbaum, de Silva, and Langford (2000).
    
    Parameters:
    X : ndarray
        NxM Data matrix with N points in the mesh for M simulations
    dims: int
        Embedding dimensionality to use 
    n_size : int
        Neighborhood size (number of neighbors for 'k' method)
    comp: int
        Component to embed, if more than 1 (defaults to 1, the largest)
    verbose: bool
        Display information (default is True)
    
    Returns:
    Y : ndarray
        Contains coordinates for d-dimensional embeddings in Y.
    R : list
        Residual variances for the embedding in Y.
    E : ndarray
        Edge matrix for neighborhood graph.
    """    
    # Compute pairwise distances in a condensed form and convert to a square form
    D = cp.asnumpy(euclidean_d(X)) if type(X) is cp.ndarray else euclidean_d(X)

    # Step 0: Initialization and Parameters
    N = D.shape[0]
    if D.shape[1] != N:
        raise ValueError("D must be a square matrix")

    K = n_size
    if not isinstance(K, int) or K <= 0:
        raiseError("Number of neighbors for 'k' method must be a positive integer")
    
    INF = 1000 * np.max(D) * N  # Effectively infinite distance
        
    # Step 1: Construct neighborhood graph
    if verbose:
        pprint(0,"Constructing neighborhood graph...",flush=True)
        
    # Sort distances and keep only K-nearest neighbors
    sorted_indices = np.argsort(D, axis=1)
    for i in range(N):
        D[i, sorted_indices[i, K+1:]] = INF  # Only keep the K nearest neighbors

    D = np.minimum(D, D.T)  # Ensure distance matrix is symmetric

    E = (D != INF).astype(int)  # Edge matrix for neighborhood graph

    # Step 2: Compute shortest paths using Floyd-Warshall algorithm
    if verbose:
        pprint(0,"Computing shortest paths...",flush=True)

    G = shortest_path(D, method='FW', directed=False, unweighted=False)

    # Step 3: Identify connected components
    firsts = np.min((G == INF).astype(int) * np.arange(N), axis=1)
    comps, indices = np.unique(firsts, return_inverse=True)
    size_comps = np.array([np.sum(indices == i) for i in range(len(comps))])
    
    sorted_indices = np.argsort(size_comps)[::-1]
    comps = comps[sorted_indices]
    size_comps = size_comps[sorted_indices]

    if comp > len(comps):
        comp = 1  # Default to largest component if specified component is out of range

    if verbose:
        pprint(0,f"Number of connected components in graph: {len(comps)}",flush=True)
        pprint(0,f"Embedding component {comp} with {size_comps[comp-1]} points.",flush=True)

    # Select the largest connected component
    index = np.where(firsts == comps[comp-1])[0]
    G = G[np.ix_(index, index)]
    N = len(index)

    # Step 4: Construct low-dimensional embeddings using Classical MDS
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ (G**2) @ H  # Centering the matrix
    
    eigenvalues, eigenvectors = eigh(B, subset_by_index=[N - dims, N - 1])
    
    # Sorting eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute embeddings for each specified dimension
    if dims <= N:
        coords = eigenvectors[:, :dims] * np.sqrt(eigenvalues[:dims])
        Y = coords.T
        
        # Compute residual variance
        L2_D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2).flatten()
        r2 = 1 - np.corrcoef(L2_D, G.flatten())[0, 1]**2
        R = r2

        if verbose:
            pprint(0,f"Isomap on {N} points with dimensionality {dims} --> residual variance = {R}",flush=True)
    else:
        raiseError('Selected number of dimensions is higher than the number of samples')
        return None, None, None

    return Y, R, E


@cr('MANIFOLD.mds')
def mds(X:np.ndarray, dims:int, verbose:bool = True):
    """
    Computes the MDS embedding using a custom approach with squared distances and eigen-decomposition.
    
    Parameters:
    X : ndarray
        NxM Data matrix with N points in the mesh for M simulations
    dims : int
        Embedding dimensionality to use (p in your MATLAB code)
    
    Returns:
    Y : ndarray
        Contains coordinates for d-dimensional embeddings in Y.
    """
    # Step 1: Compute the pairwise Euclidean distance matrix and square it
    D = cp.asnumpy(euclidean_d(X)) if type(X) is cp.ndarray else euclidean_d(X)
    D = D * D

    # Step 2: Apply the custom centering formula to get matrix B
    n = D.shape[0]
    C = np.eye(n) - np.ones((n,n))/n
    B = -0.5*np.matmul(np.matmul(C,D),C)

    # Step 3: Compute eigen-decomposition
    # eigenvalues, eigenvectors = eigh((B + B.T) / 2)
    eigenvalues, eigenvectors = eigh(B)

    # Step 4: Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[:dims]
    eigenvectors = eigenvectors[:,:dims]

    # Step 5: Keep only positive eigenvalues beyond a roundoff threshold
    threshold = np.max(np.abs(eigenvalues)) * np.finfo(eigenvalues.dtype).eps**(3 / 4)
    keep = eigenvalues > threshold
    if not np.any(keep):
        Y = np.zeros((n, 1))
    else:
        # Compute the embedding with kept eigenvalues and corresponding eigenvectors
        Y = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])

    return Y.T