#!/bin/env python
#
# Conversion from NLR7301 dataset to 
# pyLOM v3.0 format
#
# 27/09/2024
from __future__ import print_function, division

import os,glob,numpy as np, netCDF4 as NC4

import torch

import pyLOM
from pyLOM.NN import Graph


def process_edge_vectors(edge_index: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Process edge vectors from Cartesian to polar coordinates.
    ----------
    edge_index : torch.Tensor
        Edge index tensor.
    xyz : np.ndarray
        Node coordinates.
    
    Returns
    -------
    edge_vecs : np.ndarray
        Processed edge vectors in polar coordinates.
    """
    edge_vecs = np.zeros((edge_index.shape[1], 2), dtype=np.float64)  # Initialize array for edge vectors
    for edge in edge_index.T:
        c_i = xyz[edge[0]]
        c_j = xyz[edge[1]]
        d_ij = c_j - c_i
        # Transform to polar coordinates
        d_ij = np.array([np.linalg.norm(d_ij), np.arctan2(d_ij[1], d_ij[0])])
        edge_vecs[edge[0], :] = d_ij

    return edge_vecs


if __name__ == "__main__":
    DATAPATH = "/home/p.yeste/CETACEO_DATA/nlr7301/"


    ## Datasets
    DATASETS = ['test','train','val']
    NPOINTS  = 597 # According to documentation


    ## Create a serial partition table
    ptable = pyLOM.PartitionTable.new(1,NPOINTS,NPOINTS)
    print(ptable)

    
    ## Loop on the datasets
    for dset in DATASETS:
        print(dset)
        # Generate a list of the available files
        filelist = glob.glob(os.path.join(DATAPATH,dset,'*'))
        # Obtain available Mach and AoA
        case   = [int(f.split('_')[-3][4:]) for f in filelist]
        Mvec   = [float(f.split('_')[-2][1:]) for f in filelist]
        AoAvec = [float(f.split('_')[-1][3:]) for f in filelist]
        # Create input vectors
        xyz = np.zeros((NPOINTS,2),np.double)
        X   = np.zeros((NPOINTS,len(Mvec)),np.double) # assuming len(Mvec) == len(AoAvec) according to documentation
        # Read dataset and populate
        ii = 0
        for c,M,AoA in zip(case,Mvec,AoAvec):
            ncfile = NC4.Dataset(os.path.join(DATAPATH,dset,'Snap_Case%04d_M%.5f_AoA%.5f'%(c,M,AoA)))
            xyz[:,0] = ncfile.variables['x'][:597]
            xyz[:,1] = ncfile.variables['z'][:597]
            X[:,ii]  = ncfile.variables['cp'][:597]
            ncfile.close()
            ii += 1
        # Create a pyLOM dataset
        d = pyLOM.Dataset(xyz=xyz, ptable=ptable, order=np.arange(xyz.shape[0]), point=True,
            # Add the variables
            vars  = {
                'Mach':{'idim':0,'value':np.array(Mvec)},
                'AoA' :{'idim':0,'value':np.array(AoAvec)},
            },
            # Now add all the arrays to be stored in the dataset
            # It is important to convert them as C contiguous arrays
            # DUMMYVAR = {'ndim':3, 'value':dummy_var},
            CP      = {'ndim':1,'value':X},
        )
        print(d)
        d.save(f'{DATAPATH+dset.upper()}_converter.h5', append=False) # Store dataset

    
    # Create a pyLOM Graph and append it to the datasets (used by GNS)
    ncfile = NC4.Dataset(os.path.join(DATAPATH,'train','Snap_Case0000_M0.52500_AoA-0.33333'))
    xyz = np.zeros((597,2), np.double)  # Assuming 597 points as per documentation
    xyz[:,0] = ncfile.variables['x'][:597]
    xyz[:,1] = ncfile.variables['z'][:597]

    ## Load surface normals
    normals = np.load(DATAPATH+"normals.npz")["normals"]
    print("normals:", normals.shape)

    ## Load element wall normals
    wall_normals = np.load(DATAPATH+"faceNormals.npz")["faceNormals"]
    print("wall normals:", wall_normals.shape)

    ## Load edges in COO format
    edges_coo = np.load(DATAPATH+"edgesCOO.npz")["edgesCOO"]
    print("edges COO:", edges_coo.shape)

    edge_index = torch.tensor(edges_coo, dtype=torch.long)  # Convert to torch tensor
    node_attrs = {
        'xyz': torch.tensor(xyz, dtype=torch.float),
        'normals': torch.tensor(normals, dtype=torch.float),
    }

    edge_vecs = process_edge_vectors(edge_index, xyz)
    edge_attrs = {
        'edge_vecs': torch.tensor(edge_vecs, dtype=torch.float),
        'wall_normals': torch.tensor(wall_normals, dtype=torch.float),
    }

    g = Graph(edge_index=edge_index, node_attrs=node_attrs, edge_attrs=edge_attrs)
    print(g)

    # Append the graph to the h5 files
    for dset in DATASETS:
        path = f'{DATAPATH+dset.upper()}_converter.h5'
        print(f"Appending graph to {path}")
        g.save(path, mode='a')  # Append the graph to the existing dataset
