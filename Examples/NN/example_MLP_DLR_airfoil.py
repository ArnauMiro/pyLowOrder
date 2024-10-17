#!/usr/bin/env python
#
# Example of MLP.
#
# Last revision: 09/10/2024

import os, numpy as np, torch, optuna
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


## Set device
device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Load datasets and set up the results output
BASEDIR = './DATA'
CASESTR = 'NRL7301'
RESUDIR = 'MLP_DLR_airfoil'
pyLOM.NN.create_results_folder(RESUDIR)

scaler     = pyLOM.NN.MinMaxScaler()
_,td_train = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_TRAIN.h5'),scaler,scaler)
_,td_test  = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_TEST.h5'),scaler,scaler)
_,td_val   = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_VAL.h5'),scaler,scaler)

print_dset_stats('train',td_train)
print_dset_stats('test', td_test)
print_dset_stats('val',  td_val)


## define the optimizer
optimizer = pyLOM.NN.OptunaOptimizer(
    optimization_params={
        "lr": (0.00001, 0.01),  # fixed parameter
        "n_layers": (1, 4),  # optimizable parameter,
        "batch_size": (128, 512),
        "hidden_size": (200, 400),
        "p_dropouts": (0.1, 0.5),
        "epochs": 50,
    },
    n_trials=25,
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
    save_dir=None,
)

model = pyLOM.NN.MLP(
    input_size=4,
    output_size=1,
    hidden_size=256,
    n_layers=2,
    p_dropouts=0.15,
)

training_params = {
    "epochs": 150,
    'lr_scheduler_step': 1,
    "optimizer_class": torch.optim.Adam,
    "loss_fn": torch.nn.MSELoss(),
    "print_rate_epoch": 1,
    "num_workers": 6,
    "device": device,
    "lr": 0.0008380427541690664, 
    "lr_gamma": 0.9905178804615045, 
    "batch_size": 119, 
    "hidden_size": 129,
    "n_layers": 6
}

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    valid_dataset=td_val,
    # To optimize the hyperparameters:
    # optimizer=optimizer,
    # model_class=pyLOM.NN.MLP,
    # To train a model:
    model=model,
    training_params=training_params,
)

pipeline.run()


## check saving and loading the model
pipeline.model.save(os.path.join(RESUDIR,"model.pth"))
model = pyLOM.NN.MLP.load(RESUDIR + "/model.pth")
# to predict from a dataset
preds = model.predict(td_test, batch_size=2048)
# to predict from a tensor
# preds = model(torch.tensor(dataset_test[:][0], device=model.device)).cpu().detach().numpy()
scaled_preds = scaler.inverse_transform([preds])[0]
scaled_y     = scaler.inverse_transform([td_test[:][1]])[0]

# check that the scaling is correct
pyLOM.pprint(0,scaled_y.min(), scaled_y.max())

pyLOM.pprint(0,f"MAE: {np.abs(scaled_preds - np.array(scaled_y)).mean()}")
pyLOM.pprint(0,f"MRE: {np.abs(scaled_preds - np.array(scaled_y)).mean() / abs(np.array(scaled_y).mean() + 1e-6)}")
pyLOM.pprint(0,f"MSE: {((scaled_preds - np.array(scaled_y)) ** 2).mean()}")

pyLOM.cr_info()