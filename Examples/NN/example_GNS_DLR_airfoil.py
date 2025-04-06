#!/usr/bin/env python
#
# Example of GNS.
#
# Last revision: 6/04/2025

import os
import numpy as np
import h5py

import torch
import torch_geometric
import matplotlib.pyplot as plt

import pyLOM
import pyLOM.NN
from pyLOM.NN import GNS, Dataset

DATA_DIR = '/home/p.yeste/CETACEO_DATA/nlr7301/'

# Load the necessary data
files = ['train', 'val', 'test']

op = {} # Operational parameters (features)
y = {} # Target values (Cp)
for file in files:
    op[file] = None
    y[file] = None
    with h5py.File(DATA_DIR + file + '.h5', 'r') as f:
        for feature in f['features']:
            if feature != 'cp': # Only parse the operational parameters
                if op[file] is None:
                    op[file] = np.array(f['features'][feature])
                else:
                    op[file] = np.concatenate((op[file], f['features'][feature]), axis=1)
            else:
                # Save the rows of the Cp values
                y[file] = np.array(f['features'][feature])
                # for row in f['features'][feature]:
                #     y[file].append(row.reshape(-1, 1))
else:
    with h5py.File(DATA_DIR + file + '.h5', 'r') as f:
        mesh_data = {}
        for data in f['mesh']:
            if data == 'edgesCOO':
                mesh_data[data] = torch.tensor(f['mesh'][data], dtype=torch.long).transpose(0, 1)
            else:
                mesh_data[data] = np.array(f['mesh'][data])

# Delete y coordinate as it is not used
mesh_data['facenormals'] = mesh_data['facenormals'][:, [0, 2]]
mesh_data['normals'] = mesh_data['normals'][:, [0, 2]]
mesh_data['xyz'] = mesh_data['xyz'][:, [0, 2]]

print(mesh_data['edgesCOO'].shape)
print(y['train'].shape)
print(mesh_data)

# Create the graph object
g = 

