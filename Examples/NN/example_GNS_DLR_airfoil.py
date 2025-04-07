#!/usr/bin/env python
#
# Example of GNS.
#
# Last revision: 6/04/2025

import numpy as np
import h5py

import torch
import optuna

from pyLOM.NN import GNS, Dataset, pyLOMGraph, MinMaxScaler, OptunaOptimizer


def load_graph_data(file_list):
    """
    Load the graph data from the specified files.

    Parameters
    ----------
    file_list : list
        List of files to load data from.
    Returns
    -------
    -------
    op : dict
        Dictionary containing operational parameters (features).
    y : dict
        Dictionary containing target values (Cp).
    mesh_data : dict
        Dictionary containing mesh data (edgesCOO, xyz, normals, facenormals).
    -------
    ------- 
    """

    op = {} # Operational parameters (features)
    y = {} # Target values (Cp)
    for file in file_list:
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
    else:
        with h5py.File(DATA_DIR + file + '.h5', 'r') as f:
            mesh_data = {}
            for data in f['mesh']:
                if data == 'edgesCOO':
                    mesh_data[data] = torch.tensor(f['mesh'][data], dtype=torch.long).transpose(0, 1)
                else:
                    mesh_data[data] = np.array(f['mesh'][data])

    return op, y, mesh_data


def process_edge_attr(edge_index, xyz, facenormals):
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
    edge_attr = np.zeros((edge_index.shape[1], 4))
    for p, edge in enumerate(edge_index.T):
        c_i = xyz[edge[0]]
        c_j = xyz[edge[1]]
        d_ij = c_j - c_i
        f_ij = facenormals[p]

        # Transform to polar coordinates
        d_ij = np.array([np.linalg.norm(d_ij), np.arctan2(d_ij[1], d_ij[0])])
        f_ij = np.array([np.linalg.norm(f_ij), np.arctan2(f_ij[1], f_ij[0])])
        
        edge_attr[p,:] = np.concatenate((d_ij, f_ij))

    return edge_attr



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DATA_DIR = '/home/p.yeste/CETACEO_DATA/nlr7301/'
    SAVE_DIR = '/home/p.yeste/CETACEO_RESULTS/nlr7301/'

    # Build the graph object

    # Load the necessary data
    files = ['train', 'val', 'test']
    
    op_params, y, mesh_data = load_graph_data(files)

    # Delete y coordinate as it is not used
    mesh_data['facenormals'] = mesh_data['facenormals'][:, [0, 2]]
    mesh_data['normals'] = mesh_data['normals'][:, [0, 2]]
    mesh_data['xyz'] = mesh_data['xyz'][:, [0, 2]]

    print(mesh_data['edgesCOO'].shape)
    print(y['train'].shape)
    print(mesh_data)

    # Create the graph object
    edge_attr = process_edge_attr(mesh_data['edgesCOO'], mesh_data['xyz'], mesh_data['facenormals'])

    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)
    xyz = scaler.fit_transform(mesh_data['xyz'])
    facenormals = scaler.fit_transform(mesh_data['facenormals'])

    xyz = torch.tensor(xyz, dtype=torch.float32)
    facenormals = torch.tensor(facenormals, dtype=torch.float32)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    # Create the graph object
    g = pyLOMGraph(
        pos = xyz,
        facenormals = facenormals,
        edge_attr = edge_attr,
        edge_index = mesh_data['edgesCOO']
    )


    # Create the datasets
    train_dataset = Dataset(
        variables_in = op_params['train'],
        variables_out = y['train'],
        inputs_scaler = scaler,
        outputs_scaler = None
    )

    test_dataset = Dataset(
        variables_in = op_params['test'],
        variables_out = y['test'],
        inputs_scaler = scaler,
        outputs_scaler = None
    )

    val_dataset = Dataset(
        variables_in = op_params['val'],
        variables_out = y['val'],
        inputs_scaler = scaler,
        outputs_scaler = None
    )

    # Create a dict with training parameters to be optimized
    optimization_params = {
        'graph': g,
        'input_dim': 2,
        'output_dim': 1,
        'latent_dim': (1, 64),
        'hidden_size': (64, 1024),
        'num_gnn_layers': (1, 16),
        'encoder_hidden_layers': (1, 16),
        'decoder_hidden_layers': (1, 16),
        'message_hidden_layers': (1, 16),
        'update_hidden_layers': (1, 16),
        'activation': torch.nn.ELU(),
        'p_dropouts': (0.0, 0.5),
        'device': device,

        'epochs': 1000,
        'lr': (1e-5, 1e-2),
        'lr_gamma': (0.9, 0.999),
        'lr_scheduler_step': 1,
        'loss_fn': torch.nn.MSELoss(reduction='mean'),
        'optimizer': torch.optim.Adam,
        'scheduler': torch.optim.lr_scheduler.StepLR,

        'batch_size': (1, 32),
        'node_batch_size': (g.num_nodes//10000, g.num_nodes//1000),
        'num_workers': 10,
        'pin_memory': True
    }


    # Define the optimizer
    optimizer = OptunaOptimizer(
        study_name = 'GNS_airfoil',
        optimization_params = optimization_params,
        n_trials = 100,
        direction = 'minimize',
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 10,
            n_warmup_steps = 10,
            interval_steps = 5
        ),
        save_dir = SAVE_DIR
    )


    # Create the optimized model
    model, optimization_params = GNS.create_optimized_model(
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        val_dataset = val_dataset,
        training_params = optimization_params,
        optimizer = optimizer,
        model_class = GNS
    )

    # Fit the model
    model.fit(train_dataset, val_dataset, **optimization_params)

    # Save the model
    model.save_model(SAVE_DIR + 'GNS_DLR.pt')

