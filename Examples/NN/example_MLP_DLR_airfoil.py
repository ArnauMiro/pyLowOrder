#!/usr/bin/env python
#
# Example of MLP.
#
# Last revision: 14/11/2024

import os, numpy as np, torch, matplotlib.pyplot as plt
import pyLOM, pyLOM.NN

seed = 19

def load_dataset(fname,inputs_scaler,outputs_scaler):
    '''
    Auxiliary function to load a dataset into a pyLOM
    NN dataset
    '''
    return pyLOM.NN.Dataset.load(
        fname,
        field_names=["CP"],
        add_mesh_coordinates=True,
        variables_names=["AoA", "Mach"],
        inputs_scaler=inputs_scaler,
        outputs_scaler=outputs_scaler,
    )

def print_dset_stats(name,td):
    '''
    Auxiliary function to print the statistics
    of a NN dataset
    '''
    x, y = next(iter(torch.utils.data.DataLoader(td, batch_size=len(td))))
    pyLOM.pprint(0,f'name={name} ({len(td)}), x ({x.shape}) = [{x.min(dim=0)},{x.max(dim=0)}], y ({y.shape}) = [{y.min(dim=0)},{y.max(dim=0)}]')

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

## Set device
device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Load datasets and set up the results output
BASEDIR = './DATA'
CASESTR = 'NRL7301'
RESUDIR = 'MLP_DLR_airfoil'
pyLOM.NN.create_results_folder(RESUDIR)

input_scaler  = pyLOM.NN.MinMaxScaler()
output_scaler = pyLOM.NN.MinMaxScaler()
td_train = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_TRAIN.h5'),input_scaler,output_scaler)
td_test  = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_TEST.h5'),input_scaler,output_scaler)
td_val   = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_VAL.h5'),input_scaler,output_scaler)

# if we want to split by flight conditions instead of using the provided split, we can do the following
dataset = td_train + td_test + td_val
generator = torch.Generator().manual_seed(seed) # set seed for reproducibility
td_train, td_test, td_val = dataset.get_splits([0.7, 0.15, 0.15], shuffle=True, generator=generator, return_views=True)

print_dset_stats('train',td_train)
print_dset_stats('test', td_test)
print_dset_stats('val',  td_val)

model = pyLOM.NN.MLP(
    input_size=4, # x, y, AoA, Mach
    output_size=1, # CP
    hidden_size=129,
    n_layers=6,
    p_dropouts=0.15,
)

training_params = {
    "epochs": 50,
    'lr_scheduler_step': 1,
    "optimizer_class": torch.optim.Adam,
    "loss_fn": torch.nn.MSELoss(),
    "print_rate_epoch": 5,
    "num_workers": 0,
    "device": device,
    "lr": 0.000838, 
    "lr_gamma": 0.99, 
    "batch_size": 119, 
}

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    valid_dataset=td_val,
    model=model,
    training_params=training_params,
)
training_logs = pipeline.run()

# check saving and loading the model
pipeline.model.save(os.path.join(RESUDIR,"model.pth"))
model = pyLOM.NN.MLP.load(RESUDIR + "/model.pth")

# check saving and loading the scalers
input_scaler.save(os.path.join(RESUDIR,"input_scaler.json"))
output_scaler.save(os.path.join(RESUDIR,"output_scaler.json"))
input_scaler = pyLOM.NN.MinMaxScaler.load(os.path.join(RESUDIR,"input_scaler.json"))
output_scaler = pyLOM.NN.MinMaxScaler.load(os.path.join(RESUDIR,"output_scaler.json"))

# to predict from a dataset
preds = model.predict(td_test, batch_size=2048)

# to predict from a tensor
# preds = model(torch.tensor(dataset_test[:][0], device=model.device)).cpu().detach().numpy()

scaled_preds = output_scaler.inverse_transform(preds)
scaled_y     = output_scaler.inverse_transform(td_test[:][1])

# check that the scaling is correct
pyLOM.pprint(0,scaled_y.min(), scaled_y.max())

evaluator = pyLOM.NN.RegressionEvaluator()
evaluator(scaled_y, scaled_preds)
evaluator.print_metrics()

true_vs_pred_plot(scaled_y, scaled_preds, RESUDIR + '/true_vs_pred.png')
plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/train_test_loss.png')

pyLOM.cr_info()
plt.show()