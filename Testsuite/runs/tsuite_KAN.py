#!/usr/bin/env python
#
# PYLOM Testsuite
# Run KAN on the synthetic dataset
#
# Last revision: 08/01/2025
from __future__ import print_function, division

import sys, os

import torch
import pyLOM, pyLOM.NN


DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]


## Set device
device = pyLOM.NN.select_device('cpu')


## Data loading
d  = pyLOM.Dataset.load(DATAFILE)
y  = d.X(*VARIABLES)
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

sample_input, sample_output = td_train[0]

## Generate model
training_params = {
    "epochs": 5,
    "lr": 1e-5,
    "optimizer_class": torch.optim.Adam,
    "lr_kwargs":{
        "gamma": 0.95,
        "step_size": len(td_train) // 8
    },
    'batch_size': 8,
    "print_eval_rate": 1,
    "verbose":False,
    "save_logs_path":RESUDIR,
}

model = pyLOM.NN.KAN(
    input_size=sample_input.shape[0],
    output_size=sample_output.shape[0],
    hidden_size=31,
    n_layers=3,
    p_dropouts=0.0,
    layer_type=pyLOM.NN.ChebyshevLayer,
    model_name="kan_example",
    device=device,
    verbose=False,
    degree=7
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
model = pyLOM.NN.KAN.load(RESUDIR + "/model.pth")
preds = model.predict(td_test, batch_size=250)

scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y     = output_scaler.inverse_transform([td_test[:][1].numpy()])[0]


## Testsuite output
pyLOM.pprint(0,'TSUITE y            =',y.min(),y.max(),y.mean())
#pyLOM.pprint(0,'TSUITE scaled_y     =',scaled_y.min(),scaled_y.max(),scaled_y.mean())
#pyLOM.pprint(0,'TSUITE scaled_preds =',scaled_preds.min(),scaled_preds.max(),scaled_preds.mean())

pyLOM.cr_info()
pyLOM.pprint(0,'End of output')