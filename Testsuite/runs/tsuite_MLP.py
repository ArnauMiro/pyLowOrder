#!/usr/bin/env python
#
# PYLOM Testsuite
# Run MLP on the synthetic dataset
#
# Last revision: 23/10/2024
from __future__ import print_function, division

import sys, os,torch
import pyLOM, pyLOM.NN


DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]


## Set device
device = pyLOM.NN.select_device('cpu')


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
y = d.X(*VARIABLES)
print(d)


## Load pyLOM dataset and set up results output
RESUDIR = os.path.join(OUTDIR,f'MLP_{DATAFILE}')
pyLOM.NN.create_results_folder(RESUDIR,verbose=False)


## Generate torch dataset
input_scaler  = pyLOM.NN.MinMaxScaler()
output_scaler = pyLOM.NN.MinMaxScaler()

dataset = pyLOM.NN.Dataset(
    variables_out       = (y,), 
    variables_in        = d.xyz,
    parameters          = [d.get_variable('Re'), d.get_variable('AoA')],
    inputs_scaler       = input_scaler,
    outputs_scaler      = output_scaler,
    snapshots_by_column = True
)

td_train, td_test = dataset.get_splits([0.8, 0.2])


## Generate model
training_params = {
    "epochs": 50,
    "lr": 0.00015,
    "lr_gamma": 0.98,
    "lr_scheduler_step": 15,
    "batch_size": 512,
    "loss_fn": torch.nn.MSELoss(),
    "optimizer_class": torch.optim.Adam,
    "print_rate_epoch": 10,
}

sample_input, sample_output = td_train[0]
model = pyLOM.NN.MLP(
    input_size=sample_input.shape[0],
    output_size=sample_output.shape[0],
    hidden_size=32,
    n_layers=2,
    p_dropouts=0.1,
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    model=model,
    training_params=training_params,
)

training_logs = pipeline.run()


## check saving and loading the model
pipeline.model.save(os.path.join(RESUDIR,"model.pth"))
model = pyLOM.NN.MLP.load(RESUDIR + "/model.pth")
preds = model.predict(td_test, batch_size=250)

scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y     = output_scaler.inverse_transform([td_test[:][1].numpy()])[0]


## Testsuite output
pyLOM.pprint(0,'TSUITE y            =',y.min(),y.max(),y.mean())
#pyLOM.pprint(0,'TSUITE scaled_y     =',scaled_y.min(),scaled_y.max(),scaled_y.mean())
#pyLOM.pprint(0,'TSUITE scaled_preds =',scaled_preds.min(),scaled_preds.max(),scaled_preds.mean())

pyLOM.cr_info()
pyLOM.pprint(0,'End of output')