#!/bin/env python
#
# Conversion from NLR7301 dataset to 
# pyLOM v3.0 format
#
# 27/09/2024
import os
import glob
import numpy as np
import netCDF4 as NC4
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import pyLOM
from pyLOM.NN import Graph


def process_edge_vectors(edge_index: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Process edge vectors from Cartesian to polar coordinates.
    """
    edge_vecs = np.zeros((edge_index.shape[1], 2), dtype=np.float64)
    for i, edge in enumerate(edge_index.T):
        c_i = xyz[edge[0]]
        c_j = xyz[edge[1]]
        d_ij = c_j - c_i
        edge_vecs[i, :] = [np.linalg.norm(d_ij), np.arctan2(d_ij[1], d_ij[0])]
    return edge_vecs


def convert_dataset(dset: str, datapath: str, npoints: int, ptable: pyLOM.PartitionTable) -> None:
    print(f"[{dset.upper()}] Processing dataset")
    folder = os.path.join(datapath, dset)
    filelist = glob.glob(os.path.join(folder, '*'))
    
    Mvec = [float(f.split('_')[-2][1:]) for f in filelist]
    AoAvec = [float(f.split('_')[-1][3:]) for f in filelist]
    case = [int(f.split('_')[-3][4:]) for f in filelist]

    xyz = np.zeros((npoints, 2), np.double)
    X = np.zeros((npoints, len(Mvec)), np.double)

    for ii, (c, M, AoA) in enumerate(tqdm(zip(case, Mvec, AoAvec), total=len(Mvec), desc=f"[{dset}]")):
        fname = os.path.join(folder, f"Snap_Case{c:04d}_M{M:.5f}_AoA{AoA:.5f}")
        with NC4.Dataset(fname) as ncfile:
            xyz[:, 0] = ncfile.variables['x'][:npoints]
            xyz[:, 1] = ncfile.variables['z'][:npoints]
            X[:, ii] = ncfile.variables['cp'][:npoints]

    d = pyLOM.Dataset(
        xyz=xyz,
        ptable=ptable,
        order=np.arange(npoints),
        point=True,
        vars={
            'Mach': {'idim': 0, 'value': np.array(Mvec)},
            'AoA': {'idim': 0, 'value': np.array(AoAvec)},
        },
        CP={'ndim': 1, 'value': X},
    )
    out_path = os.path.join(datapath, f"{dset.upper()}_converter.h5")
    d.save(out_path, append=False)
    print(f"[{dset.upper()}] Saved to {out_path}\n")
    return d


def create_graph(datapath: str, npoints: int) -> Graph:
    sample_file = os.path.join(datapath, 'train', 'Snap_Case0000_M0.52500_AoA-0.33333')
    with NC4.Dataset(sample_file) as ncfile:
        xyz = np.zeros((npoints, 2), np.double)
        xyz[:, 0] = ncfile.variables['x'][:npoints]
        xyz[:, 1] = ncfile.variables['z'][:npoints]

    normals = np.load(os.path.join(datapath, "normals.npz"))['normals']
    wall_normals = np.load(os.path.join(datapath, "faceNormals.npz"))['faceNormals']
    edge_index = torch.tensor(np.load(os.path.join(datapath, "edgesCOO.npz"))['edgesCOO'], dtype=torch.long)

    x_dict = {
        'xyz': torch.tensor(xyz, dtype=torch.float),
        'normals': torch.tensor(normals, dtype=torch.float),
    }
    edge_vecs = process_edge_vectors(edge_index.numpy(), xyz)
    edge_attr_dict = {
        'edge_vecs': torch.tensor(edge_vecs, dtype=torch.float),
        'wall_normals': torch.tensor(wall_normals, dtype=torch.float),
    }

    g = Graph(edge_index=edge_index, x_dict=x_dict, edge_attr_dict=edge_attr_dict)
    print("Graph created.")
    return g

def plot_graph_cp(x: np.ndarray, z: np.ndarray, normals: np.ndarray, cp: np.ndarray, mach: float, aoa: float, savepath: os.path) -> None:
    """ Plot the NLR7301 airfoil, along with surface normals and CP distribution for a given Mach and AoA.
    Args:
        x (np.ndarray): X-coordinates of the airfoil.
        z (np.ndarray): Z-coordinates of the airfoil.
        normals (np.ndarray): Surface normals at each point.
        cp (np.ndarray): CP distribution at each point.
        mach (float): Mach number for the case.
        aoa (float): Angle of attack for the case.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 6))
    plt.scatter(x, z, s=10)
    plt.plot(x, cp, label='CP', color='blue')
    plt.quiver(x, z, normals[:, 0], normals[:, 1], color='red', scale=10, label='Normals')
    plt.title(f'CP Distribution (Mach: {mach}, AoA: {aoa})')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(savepath,f"cp_plot_mach{mach}_aoa{aoa}.png"), dpi=300, bbox_inches='tight')
    plt.show(block=True)


def main():
    datapath = "/home/p.yeste/CETACEO_DATA/nlr7301/"
    datasets = ['test', 'train', 'val']
    npoints = 597
    ptable = pyLOM.PartitionTable.new(1, npoints, npoints)

    for dset in datasets:
        saved_dset = convert_dataset(dset, datapath, npoints, ptable)

    g = create_graph(datapath, npoints)
    for dset in datasets:
        path = os.path.join(datapath, f"{dset.upper()}_converter.h5")
        print(f"Appending graph to {path}")
        g.save(path, mode='a')

    # Select a snapshot from the dataset and plot the airfoil with CP distribution
    i_case = 0  # First case for demonstration 
    xyz = g.xyz
    x = xyz[:, 0]
    z = xyz[:, 1]
    normals = g.normals
    cp = saved_dset.fields['CP']['value'][:, i_case].flatten()
    mach = saved_dset.get_variable('Mach')[i_case]
    aoa = saved_dset.get_variable('AoA')[i_case]

    plot_graph_cp(x, z, normals, cp, mach, aoa, savepath=datapath)



if __name__ == "__main__":
    main()
