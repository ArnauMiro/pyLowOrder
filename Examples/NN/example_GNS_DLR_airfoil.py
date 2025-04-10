#!/usr/bin/env python
#
# Example of GNS.
#
# Last revision: 6/04/2025

#%%
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

import torch
import optuna

import pyLOM
from pyLOM.NN import GNS, Dataset, pyLOMGraph, MinMaxScaler, OptunaOptimizer, Pipeline

#%%
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
        with h5py.File(BASEDIR + file + '.h5', 'r') as f:
            for feature in f['features']:
                if feature != 'cp': # Only parse the operational parameters
                    if op[file] is None:
                        op[file] = np.array(f['features'][feature])
                    else:
                        op[file] = np.concatenate((op[file], f['features'][feature]), axis=1)
                else:
                    # Save the rows of the Cp values
                    y[file] = np.array(f['features'][feature]).T
    else:
        with h5py.File(BASEDIR + file + '.h5', 'r') as f:
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


def true_vs_pred_plot(y_true, y_pred, path):
    """
    Auxiliary function to plot the true vs predicted values
    """
    num_plots = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):
        plt.subplot(num_plots, 1, j + 1)
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.5)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"Scatterplot for Component {j+1}")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=300)

def plot_train_test_loss(train_loss, test_loss, path):
    """
    Auxiliary function to plot the training and test loss
    """
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    total_epochs = len(test_loss) # test loss is calculated at the end of each epoch
    total_iters = len(train_loss) # train loss is calculated at the end of each iteration/batch
    iters_per_epoch = total_iters // total_epochs
    plt.plot(np.arange(iters_per_epoch, total_iters+1, step=iters_per_epoch), test_loss, label="Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=300)


#%%
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    BASEDIR = '/home/p.yeste/CETACEO_DATA/nlr7301/'
    RESUDIR = '/home/p.yeste/CETACEO_RESULTS/nlr7301/'

    # Build the graph object

    # Load the necessary data
    files = ['train', 'val', 'test']
    
    op_params, y, mesh_data = load_graph_data(files)
    
    # Delete y coordinate as it is not used
    mesh_data['facenormals'] = mesh_data['facenormals'][:, [0, 2]]
    mesh_data['normals'] = mesh_data['normals'][:, [0, 2]]
    mesh_data['xyz'] = mesh_data['xyz'][:, [0, 2]]
    

    # Create the graph object
    edge_attr = process_edge_attr(mesh_data['edgesCOO'], mesh_data['xyz'], mesh_data['facenormals'])

    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)
    xyz = scaler.fit_transform(mesh_data['xyz'])
    unit_norms = mesh_data['normals'] / np.linalg.norm(mesh_data['normals'], axis=1).reshape(-1,1)

    xyz = torch.tensor(xyz, dtype=torch.float32)
    unit_norms = torch.tensor(unit_norms, dtype=torch.float32)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    # Create the graph object
    g = pyLOMGraph(
        pos = xyz,
        surf_norms = unit_norms,
        edge_attr = edge_attr,
        edge_index = mesh_data['edgesCOO']
    )

    scaler.fit(np.concatenate((op_params['train'], op_params['val'], op_params['test']), axis=0))

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

    eval_dataset = Dataset(
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
        'latent_dim': (1, 32),
        'hidden_size': (64, 512),
        'num_msg_passing_layers': (1, 4),
        'encoder_hidden_layers': (1, 8),
        'decoder_hidden_layers': (1, 8),
        'message_hidden_layers': (1, 8),
        'update_hidden_layers': (1, 8),
        'activation': torch.nn.ELU(),
        'p_dropouts': (0.0, 0.5),
        'device': device,

        'epochs': 30,
        'lr': (1e-5, 1e-2),
        'lr_gamma': (0.99, 0.999),
        'lr_scheduler_step': 1,
        'loss_fn': torch.nn.MSELoss(reduction='mean'),
        'optimizer': torch.optim.Adam,
        'scheduler': torch.optim.lr_scheduler.StepLR,

        'batch_size': (1, 32),
        'node_batch_size': (g.num_nodes//100, g.num_nodes//1),
        'num_workers': 1,
        'pin_memory': True
    }


    # Define the optimizer
    optimizer = OptunaOptimizer(
        optimization_params = optimization_params,
        n_trials = 100,
        direction = 'minimize',
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 5,
            n_warmup_steps = 5,
            interval_steps = 1
        ),
        save_dir = RESUDIR
    )

    pipeline = Pipeline(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    valid_dataset=eval_dataset,
    optimizer=optimizer,
    model_class=GNS
    )


    training_logs = pipeline.run()

    # check saving and loading the model
    pipeline.model.save(os.path.join(RESUDIR,"NLR7301_optuna.pth"))
    model = GNS.load(RESUDIR + "NLR7301_optuna.pth")

    # check saving and loading the scalers
    scaler.save(os.path.join(RESUDIR,"input_scaler.json"))
    input_scaler = MinMaxScaler.load(os.path.join(RESUDIR,"input_scaler.json"))

    # to predict from a dataset
    preds = model.predict(test_dataset)
    y = test_dataset[:][1]

    # to predict from a tensor
    # preds = model(torch.tensor(dataset_test[:][0], device=model.device)).cpu().detach().numpy()

    # check that the scaling is correct
    pyLOM.pprint(0,y.min(), y.max())

    evaluator = pyLOM.NN.RegressionEvaluator()
    evaluator(y, preds)
    evaluator.print_metrics()

    true_vs_pred_plot(y, preds, RESUDIR + 'true_vs_pred.png')
    plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/train_test_loss.png')

    pyLOM.cr_info()
    plt.show()


    # Compare against dlr results

    # DLR hyperparameters
    dlr_params = {
        'graph': g,
        'input_dim': 2,
        'output_dim': 1,
        'latent_dim': 16,
        'hidden_size': 256,
        'num_msg_passing_layers': 1,
        'encoder_hidden_layers': 6,
        'decoder_hidden_layers': 1,
        'message_hidden_layers': 2,
        'update_hidden_layers': 2,
        'activation': torch.nn.ELU(),
        'p_dropouts': 0.0,
        'device': device,

        'epochs': 1000,
        'lr': 6.50e-4,
        'lr_gamma': 0.9954,
        'lr_scheduler_step': 1,
        'loss_fn': torch.nn.MSELoss(reduction='mean'),
        'optimizer': torch.optim.Adam,
        'scheduler': torch.optim.lr_scheduler.StepLR,

        'batch_size': 15,
        'node_batch_size': 32,
        'num_workers': 1,
        'pin_memory': True
    }

    dlr_model = GNS(
        input_dim = 2,
        output_dim = 1,
        latent_dim = 16,
        hidden_size = 256,
        num_msg_passing_layers = 1,
        encoder_hidden_layers = 6,
        decoder_hidden_layers = 1,
        message_hidden_layers = 2,
        update_hidden_layers = 2,
        **dlr_params        
    )

    pipeline = Pipeline(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset=eval_dataset,
        model=dlr_model,
        training_params=dlr_params
    )
    training_logs = pipeline.run()
    # check saving and loading the model
    pipeline.model.save(os.path.join(RESUDIR,"NLR7301_DLR.pth"))
    model = GNS.load(RESUDIR + "NLR7301_DLR.pth")

    # to predict from a dataset
    preds = model.predict(test_dataset)
    y = test_dataset[:][1]

    evaluator = pyLOM.NN.RegressionEvaluator()
    evaluator(y, preds)
    evaluator.print_metrics()

    true_vs_pred_plot(y, preds, RESUDIR + 'true_vs_pred_DLR.png')
    plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/train_test_loss_DLR.png')

    pyLOM.cr_info()
    plt.show()

