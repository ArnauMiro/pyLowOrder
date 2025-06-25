#%%
#!/bin/env python
#
# Conversion from NLR7301 dataset to 
# pyLOM v3.0 format
#
# 27/09/2024
#%%
from __future__ import print_function, division

import os,glob,numpy as np, netCDF4 as NC4
import pyLOM

#%%
def process_edge_attr(edge_index: np.ndarray, xyz: np.ndarray, facenormals: np.ndarray) -> np.ndarray:
    """
    Process the edge attributes based on the edge index and mesh data.
    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index tensor.
    xyz : np.ndarray
        Node coordinates.
    facenormals : np.ndarray
        Face normals.
    -------
    -------
    edge_attr : np.ndarray
        Edge attributes.
    -------
    -------
    """
    for p, edge in enumerate(edge_index.T):
        c_i = xyz[edge[0]]
        c_j = xyz[edge[1]]
        d_ij = c_j - c_i
        f_ij = facenormals[p]

        # Transform to polar coordinates
        d_ij = np.array([np.linalg.norm(d_ij), np.arctan2(d_ij[1], d_ij[0])])
        f_ij = np.array([np.linalg.norm(f_ij), np.arctan2(f_ij[1], f_ij[0])])

    return d_ij

#%%
if __name__ == "__main__":
    DATAPATH = "/home/p.yeste/CETACEO_DATA/nlr7301/"


    ## Datasets
    DATASETS = ['test','train','val']
    NPOINTS  = 597 # According to documentation


    ## Create a serial partition table
    ptable = pyLOM.PartitionTable.new(1,NPOINTS,NPOINTS)
    print(ptable)

    #%%
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

    #%%
    # Create a pyLOM Graph and append it to the datasets (used by GNS)
    ncfile = NC4.Dataset(os.path.join(DATAPATH,'train','Snap_Case0000_M0.52500_AoA-0.33333'))
    xyz = np.zeros()
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


    # Compute the node and edge attributes
    node_attr = np.stack((xyz, normals), axis=1)
    print("node_attr: ", node_attr.shape)

    edge_attr = process_edge_attr(edge_index=edges_coo, xyz=xyz, facenormals=normals)
    print("edge_attr: ", edge_attr.shape)

    g = pyLOM.Graph(node_attr=node_attr, edge_index=edges_coo.T, edge_attr=edge_attr)

    # Append the graph to the dataset
    for dset in DATASETS:
        pass

#%%
import torch
import torch_geometric as pyg

pos = torch.rand(3, 2)  # Random positions for 3 nodesy
z = torch.rand(3, 1)  # Random z-coordinates for 3 nodes
edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 0]], dtype=torch.long)
edge_attr = torch.tensor([[1, 0.5, 0.2, 0.1], [1, 0.5, 0.2, 0.1], [1, 0.5, 0.2, 0.1]], dtype=torch.float)
q = torch.rand(4,2,2)

g = pyg.data.Data(pos=pos, z=z, edge_index=edge_index, edge_attr=edge_attr, name='test_graph', q=q)

# print(g.is_directed())

g.generate_ids()

print(g.node_attrs())
print(g.edge_attrs())

print(g.keys())

print(g.n_id)
print(g.e_id)
print(g.num_edges)


# %%
