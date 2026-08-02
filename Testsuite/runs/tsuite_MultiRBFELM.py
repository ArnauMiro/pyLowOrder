#!/usr/bin/env python
#
# PYLOM Testsuite
# Run RBFELM on the synthetic dataset
#
# Last revision: 18/06/2026
import sys, os, torch, optuna
import pyLOM, pyLOM.NN

DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
SEED      = 42
GEN       = torch.Generator().manual_seed(SEED)


## Set device
device = pyLOM.NN.select_device('cpu')


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
y = d.X(*VARIABLES)


## Load pyLOM dataset and set up results output
RESUDIR = os.path.join(OUTDIR,f'RBFELM_{DATAFILE}')
pyLOM.NN.create_results_folder(RESUDIR,verbose=False)


## Generate torch dataset
dataset = pyLOM.NN.Dataset(
    variables_out       = (y[:,0],), 
    variables_in        = d.xyz,
    parameters          = [d.get_variable('Re')[:1], d.get_variable('AoA')[:1]],
    inputs_scaler       = None,
    outputs_scaler      = None,
    snapshots_by_column = True
)

dataset.remove_column(3, from_variables_out=False)
dataset.remove_column(2, from_variables_out=False)

td_train, td_valid, td_test = dataset.get_splits([0.6, 0.2, 0.2], return_views=False, generator=GEN)

td_train.print_stats(dataset_name='Train Dataset')
td_valid.print_stats(dataset_name='Validation Dataset')
td_test.print_stats(dataset_name='Test Dataset')

## Generate model
optimization_params = {
    "n_clusters":      (1, 3),
    "n_centers":       (20, 50),
    "overlap_factor":  (1.0, 2.0),
    "reg_lambda":      (1e-6, 1e-1),
    "gamma_k":         (1, 5),
    "gamma_alpha":     (0.5, 2.0),
    "center_sampling": "random",
    "gamma_mode":      "local",
    "batch_size":      100_000,
    "seed":            SEED,
}

optimizer = pyLOM.NN.OptunaOptimizer(
    optimization_params = optimization_params,
    n_trials            = 10,
    direction           = "minimize",
    save_dir            = None,
    sampler             = optuna.samplers.TPESampler(seed=SEED)
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    valid_dataset=td_valid,
    test_dataset=td_test,
    model_class=pyLOM.NN.MultiRBFELM,
    optimizer=optimizer,
)

training_logs = pipeline.run()


## check saving and loading the model
pipeline.model.save(os.path.join(RESUDIR,"model.pth"), verbose=False)
model = pyLOM.NN.MultiRBFELM.load(RESUDIR + "/model.pth", verbose=False)
preds, trues = model.predict(td_test, return_targets=True)


## Testsuite output
pyLOM.pprint(0, 'TSUITE y            =',y.min(),y.max(),y.mean())
pyLOM.pprint(0, 'TSUITE y_test       =',trues.min(),trues.max(),trues.mean())
pyLOM.pprint(0, 'TSUITE y_pred       =',preds.min(),preds.max(),preds.mean())

pyLOM.cr_info()
pyLOM.pprint(0,'End of output')