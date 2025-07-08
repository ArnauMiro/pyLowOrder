#!/usr/bin/env python
#
# Example of GNS.
#
# Last revision: 6/04/2025

#%%
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import optuna

import pyLOM
from pyLOM.NN import GNS, Graph, MinMaxScaler, OptunaOptimizer, Pipeline

#%%
def load_dataset(fname,inputs_scaler,outputs_scaler):
    '''
    Auxiliary function to load a dataset into a pyLOM
    NN dataset
    '''
    return pyLOM.NN.Dataset.load(
        fname,
        field_names=["CP"],
        add_variables=True,
        add_mesh_coordinates=False,
        variables_names=["AoA", "Mach"],
        inputs_scaler=inputs_scaler,
        outputs_scaler=outputs_scaler,
        squeeze_last_dim=False,  # Keep the last dimension for outputs
    )

def print_dset_stats(name, td):
    '''
    Auxiliary function to print the statistics of a NN dataset
    '''
    x, y = next(iter(torch.utils.data.DataLoader(td, batch_size=len(td))))
    x_min, x_max = x.min(dim=0)[0], x.max(dim=0)[0]
    y_min, y_max = y.min(dim=0)[0], y.max(dim=0)[0]
    pyLOM.pprint(0, f"{name} ({len(td)} samples):")
    pyLOM.pprint(0, f"  x: {x.shape}, range = [{x_min.tolist()}, {x_max.tolist()}]")
    pyLOM.pprint(0, f"  y: {y.shape}, range = [{y_min.tolist()}, {y_max.tolist()}]")


def true_vs_pred_plot(y_true, y_pred, path):
    """
    Auxiliary function to plot the true vs predicted values
    """
    num_plots = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):
        plt.subplot(num_plots, 1, j + 1)
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.5)
        plt.plot([y_true[:, j].min(), y_true[:, j].max()], [y_true[:, j].min(), y_true[:, j].max()], 'r--')
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
def main():
    ## Set device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = pyLOM.NN.select_device(device_name)

    ## Load datasets and set up the results output
    BASEDIR = '/home/p.yeste/CETACEO_DATA/nlr7301/'
    RESUDIR = '/home/p.yeste/CETACEO_RESULTS/nlr7301/'
    pyLOM.NN.create_results_folder(RESUDIR)

    input_scaler  = pyLOM.NN.MinMaxScaler()
    # output_scaler = pyLOM.NN.MinMaxScaler()
    output_scaler = None # No output scaling in this example
    td_train = load_dataset(os.path.join(BASEDIR,'TRAIN_converter.h5'),input_scaler,output_scaler)
    td_test  = load_dataset(os.path.join(BASEDIR,'TEST_converter.h5'),input_scaler,output_scaler)
    td_val   = load_dataset(os.path.join(BASEDIR,'VAL_converter.h5'),input_scaler,output_scaler)

    # print(td_train)
    # print(td_train[0])
    # print(td_train[1])

    # print_dset_stats("Train", td_train)
    # print_dset_stats("Test", td_test)return_loss (bool): Whether to compute and return loss instead of predictions.
    # print_dset_stats("Val", td_val)

    ## Create the graph
    g = Graph.load(os.path.join(BASEDIR, "TRAIN_converter.h5"), device=device)
    print(g)

#%%  
    ## Create and run a pipeline for optimizing a GNS model
    optimization_params = {
        # --- Model parameters ---
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

        # --- Training parameters ---
        'epochs': 3,
        'lr': (1e-5, 1e-2),
        'lr_gamma': (0.99, 0.999),
        'lr_scheduler_step': 1,
        'loss_fn': torch.nn.MSELoss(reduction='mean'),
        'optimizer': torch.optim.Adam,
        'scheduler': torch.optim.lr_scheduler.StepLR,

        # --- Loader parameters ---
        'batch_size': (1, 32),
        'node_batch_size': (g.num_nodes//100, g.num_nodes//1),
        'num_workers': 1,
        'pin_memory': True
    }


    # Define the optimizer
    optimizer = OptunaOptimizer(
        optimization_params = optimization_params,
        n_trials = 2,
        direction = 'minimize',
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 1,
            n_warmup_steps = 1,
            interval_steps = 1
        ),
        save_dir = RESUDIR
    )

    # Create the pipeline
    pipeline = Pipeline(
        train_dataset=td_train,
        test_dataset=td_test,
        valid_dataset=td_val,
        optimizer=optimizer,
        model_class=GNS
    )

    # Run the optimization
    training_logs = pipeline.run()

    # check saving and loading the model
    pipeline.model.save(os.path.join(RESUDIR,"NLR7301_optuna_test.pth"))
    model = GNS.load(RESUDIR + "NLR7301_optuna_test.pth")
    print(model)

    # check saving and loading the scalers
    if input_scaler is not None:
        input_scaler.save(os.path.join(RESUDIR,"input_scaler_test.json"))
        input_scaler = MinMaxScaler.load(os.path.join(RESUDIR,"input_scaler_test.json"))
    if output_scaler is not None:
        output_scaler.save(os.path.join(RESUDIR,"output_scaler_test.json"))
        output_scaler = MinMaxScaler.load(os.path.join(RESUDIR,"output_scaler_test.json"))

    # to predict from a dataset
    print("Predicting from the test dataset...")
    preds_train = model.predict(td_test)
    labels_train = td_test[:][1]
    preds_train = preds_train.cpu()
    labels_train = labels_train.cpu()

    # to predict from a tensor
    print("Predicting from a tensor...")
    # Example input tensor. Shape must be [B, D] where B is the batch size and D is the input dimension
    inputs_tensor = torch.tensor([[0.4, 0.7]], dtype=torch.float32) 
    preds_tensor = model.predict(inputs_tensor)
    print(f"Inputs tensor: {inputs_tensor.shape}, Predictions tensor: {preds_tensor.shape}")

    # check that the scaling is correct
    if output_scaler is not None:
        pyLOM.pprint(0,labels_train.min(), labels_train.max())

    evaluator = pyLOM.NN.RegressionEvaluator()
    evaluator(labels_train, preds_train)
    evaluator.print_metrics()

    true_vs_pred_plot(labels_train, preds_train, RESUDIR + 'true_vs_pred_test.png')
    plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/train_test_loss_test.png')

    pyLOM.cr_info()
    plt.show()


    ## Create a new pipeline to train a GNS model with the best parameters found by the DLR institute in the original GNS paper
    dlr_params = {
        # Model parameters
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

        # Training parameters
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

    # Instantiate a model with the DLR parameters
    dlr_model = GNS(**dlr_params)
    print(dlr_model)

    # Create a new pipeline for training with DLR parameters (no optimization)
    pipeline = Pipeline(
        train_dataset=td_train,
        test_dataset=td_test,
        valid_dataset=td_val,
        model=dlr_model,
        training_params=dlr_params
    )

    # Run the training
    training_logs = pipeline.run()

    # check saving and loading the model
    pipeline.model.save(os.path.join(RESUDIR,"NLR7301_DLR_test.pth"))
    model = GNS.load(RESUDIR + "NLR7301_DLR_test.pth")

    # to predict from a dataset
    preds = model.predict(td_test)
    labels = td_test[:][1]

    evaluator = pyLOM.NN.RegressionEvaluator()
    evaluator(labels, preds)
    evaluator.print_metrics()

    true_vs_pred_plot(labels, preds, RESUDIR + 'true_vs_pred_DLR_test.png')
    plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/train_test_loss_DLR_test.png')

    pyLOM.cr_info()
    plt.show()

#%%
if __name__ == "__main__":
    main()

# %%

t = torch.tensor([[0,1],[2,3]])
ds = pyLOM.NN.Dataset(variables_in=t)

print(ds)
# %%
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.tensor(np.random.rand(100, 2), dtype=torch.float32)
y = torch.tensor(np.random.rand(100, 5, 1), dtype=torch.float32)

x1 = torch.tensor(np.random.rand(2,), dtype=torch.float32)
print(x1)

ds1 = TensorDataset(x1)
dl1 = DataLoader(ds1, batch_size=1, shuffle=True)
print(f"len(ds1): {len(ds1)}")

ds = TensorDataset(x, y)
dl = DataLoader(ds, batch_size=1, shuffle=True)
print(f"len(ds): {len(ds)}")


for batch in dl:
    x_batch, y_batch = batch
    print(f"x: {x_batch.shape}, y: {y_batch.shape}")
    print(f"x: {x_batch}, y: {y_batch}")
    break  # Just to show the first batch
for batch in dl1:
    x_batch1 = batch[0]
    print(f"x1: {x_batch1.shape}")
    print(f"x1: {x_batch1}")
    break  # Just to show the first batch



# %%
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import pyLOM

datapath = '/home/p.yeste/CETACEO_DATA/'

npoints = 100
ptable = pyLOM.PartitionTable.new(1, npoints, npoints)

xyz = np.random.rand(npoints, 2).astype(np.float32)

var1 = np.random.rand(50,).astype(np.float32)
var2 = np.random.rand(50,).astype(np.float32)
outputs_1 = np.random.rand(npoints, 50, 1).astype(np.float32)
outputs_2 = np.random.rand(npoints, 50, 2).astype(np.float32)
outputs_3 = np.random.rand(npoints, 50).astype(np.float32)
outputs_4 = np.random.rand(npoints, 50).astype(np.float32)

d = pyLOM.Dataset(
    xyz=xyz,
    ptable=ptable,
    order=np.arange(npoints),
    point=True,
    vars={
        'Var1': {'idim': 0, 'value': var1},
        'Var2': {'idim': 0, 'value': var2},
    },
    OPT1={'ndim': 1, 'value': outputs_1},
    OPT2={'ndim': 2, 'value': outputs_2},
    OPT3={'ndim': 1, 'value': outputs_3},
    OPT4={'ndim': 1, 'value': outputs_4},
)

print(d)

out_path = os.path.join(datapath, "test_dataset.h5")
d.save(out_path, append=False)
print(f"Dataset saved to {out_path}")

#%%
nnd = pyLOM.NN.Dataset.load(
    out_path,
    field_names=["OPT1", "OPT2", "OPT3", "OPT4"],
    add_variables=True,
    add_mesh_coordinates=False,
    variables_names=["Var1", "Var2"],
    inputs_scaler=None,
    outputs_scaler=None,
)
print(nnd)

for i in range(len(nnd)):
    x, y = nnd[i]
    print(f"x: {x.shape}, y: {y.shape}")
    print(f"x: {x}, y: {y}")
    if i == 0:
        break  # Just to show the first sample

# %%
