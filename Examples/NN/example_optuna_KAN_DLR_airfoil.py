#!/usr/bin/env python
#
# Example of KAN with optuna.
#
# Last revision: 25/10/2024

import os, numpy as np, optuna, matplotlib.pyplot as plt
import pyLOM


def load_dataset(fname,inputs_scaler,outputs_scaler):
    '''
    Auxiliary function to load a dataset into a pyLOM
    NN dataset
    '''
    d  = pyLOM.Dataset.load(fname)
    td = pyLOM.NN.Dataset(
        variables_out  = (d["CP"],), 
        variables_in   = d.xyz,
        # to have each Mach and AoA pair just once. 
        # To have all possible combinations, use [d.get_variable('AoA'), d.get_variable("Mach")]
        parameters     = [[*zip(d.get_variable('AoA'), d.get_variable('Mach'))]], 
        inputs_scaler  = inputs_scaler,
        outputs_scaler = outputs_scaler,
    )
    return d, td

def print_dset_stats(name,td):
    '''
    Auxiliary function to print the statistics
    of a NN dataset
    '''
    x, y = td[:]
    pyLOM.pprint(0,f'name={name} ({len(td)}), x ({x.shape}) = [{x.min(dim=0)},{x.max(dim=0)}], y ({y.shape}) = [{y.min(dim=0)},{y.max(dim=0)}]')

def true_vs_pred_plot(y_true, y_pred, path):
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

## Set device
device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Load datasets and set up the results output
BASEDIR = './DATA'
CASESTR = 'NRL7301'
RESUDIR = 'KAN_optuna_DLR_airfoil'
pyLOM.NN.create_results_folder(RESUDIR)

input_scaler     = pyLOM.NN.MinMaxScaler()
output_scaler     = pyLOM.NN.MinMaxScaler()
_,td_train = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_TRAIN.h5'),input_scaler,output_scaler)
_,td_test  = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_TEST.h5'),input_scaler,output_scaler)
_,td_val   = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_VAL.h5'),input_scaler,output_scaler)

print_dset_stats('train',td_train)
print_dset_stats('test', td_test)
print_dset_stats('val',  td_val)


optimization_params = {
    "lr": (0.00001, 0.1),
    "batch_size": (10, 64),
    "hidden_size": (10, 40),
    "print_rate": 10,
    "n_layers": (1, 4),
    "print_eval_rate": 2,
    "epochs": 10,
    "lr_gamma": 0.98,
    "lr_step_size": 3,
    "model_name": "kan_test_optuna",
    'device': device,
    "layer_type": (pyLOM.NN.ChebyshevLayer, pyLOM.NN.JacobiLayer),
    "layer_kwargs": {
        "degree": (3, 10),
    },
}

# define the optimizer
optimizer = pyLOM.NN.OptunaOptimizer(
    optimization_params=optimization_params,
    n_trials=5, # 5 may be too low, but it is just for the example
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
    save_dir=None,
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    valid_dataset=td_val,
    model_class=pyLOM.NN.KAN,
    optimizer=optimizer,
)

training_logs = pipeline.run()


## check saving and loading the model
pipeline.model.save(os.path.join(RESUDIR, "model.pth"))
model = pyLOM.NN.KAN.load(RESUDIR + "/model.pth")
# to predict from a dataset
preds = model.predict(td_test, batch_size=2048)
# to predict from a tensor
# preds = model(torch.tensor(dataset_test[:][0], device=model.device)).cpu().detach().numpy()
scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y     = output_scaler.inverse_transform([td_test[:][1]])[0]

# check that the scaling is correct
pyLOM.pprint(0,scaled_y.min(), scaled_y.max())

pyLOM.pprint(0,f"MAE: {np.abs(scaled_preds - np.array(scaled_y)).mean()}")
pyLOM.pprint(0,f"MRE: {np.abs(scaled_preds - np.array(scaled_y)).mean() / abs(np.array(scaled_y).mean() + 1e-6)}")
pyLOM.pprint(0,f"MSE: {((scaled_preds - np.array(scaled_y)) ** 2).mean()}")

true_vs_pred_plot(scaled_y, scaled_preds, RESUDIR + '/true_vs_pred.png')
plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/train_test_loss.png')

pyLOM.cr_info()
plt.show()