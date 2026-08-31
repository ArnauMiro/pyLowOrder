#!/usr/bin/env python
#
# PYLOM Testsuite
# Run MulticlassClassifier on the synthetic dataset
#
# Last revision: 28/08/2026
import sys, os, numpy as np, torch, optuna
import pyLOM, pyLOM.NN

DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
SEED      = 42
N_CLASSES = 4
GEN       = torch.Generator().manual_seed(SEED)


## Set device
device = pyLOM.NN.select_device('cpu')


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
y = d.X(*VARIABLES)

## Build synthetic multiclass labels from the continuous field (quantile bins)
cp     = y[:,0]
edges  = np.quantile(cp, np.linspace(0, 1, N_CLASSES + 1)[1:-1])
labels = np.digitize(cp, edges).astype(np.float64)


## Load pyLOM dataset and set up results output
RESUDIR = os.path.join(OUTDIR,f'MulticlassClassifier_{DATAFILE}')
pyLOM.NN.create_results_folder(RESUDIR,verbose=False)


## Generate torch dataset
dataset = pyLOM.NN.Dataset(
    variables_out       = (labels,),
    variables_in        = d.xyz,
    parameters          = [d.get_variable('Re')[:1], d.get_variable('AoA')[:1]],
    inputs_scaler       = None,
    outputs_scaler      = None,
    snapshots_by_column = True
)

dataset.remove_column(3, from_variables_out=False)
dataset.remove_column(2, from_variables_out=False)


## Scale inputs
def scale_inputs(inputs, outputs, inputs_scaler):
    if inputs_scaler.is_fitted:
        inputs = inputs_scaler.transform(inputs)
    else:
        inputs = inputs_scaler.fit_transform(inputs)
    return inputs, outputs

input_scaler = pyLOM.NN.MinMaxScaler()

dataset.map(
    scale_inputs,
    fn_kwargs={"inputs_scaler": input_scaler},
    batched=True,
    batch_size=len(dataset),
)

td_train, td_valid, td_test = dataset.get_splits([0.6, 0.2, 0.2], return_views=False, generator=GEN)

td_train.print_stats(dataset_name='Train Dataset')
td_valid.print_stats(dataset_name='Validation Dataset')
td_test.print_stats(dataset_name='Test Dataset')


## Generate model
optimization_params = {
    "n_estimators":  (10, 50),
    "max_depth":     (2, 4),
    "learning_rate": (0.05, 0.3),
    "seed":          SEED,
    "n_classes":     N_CLASSES,
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
    model_class=pyLOM.NN.MulticlassClassifier,
    optimizer=optimizer,
)

training_logs = pipeline.run()


## check saving and loading the model
pipeline.model.save(os.path.join(RESUDIR,"model.pth"))
model = pyLOM.NN.MulticlassClassifier.load(RESUDIR + "/model.pth", verbose=False)
preds, trues = model.predict(td_test, return_targets=True)


## Performance check: fraction of misclassified test points
pred_classes = preds.argmax(axis=1)
mean_error   = 1.0 - (pred_classes == trues).mean()


## Testsuite output
pyLOM.pprint(0,'TSUITE y            =',labels.min(),labels.max(),labels.mean())
pyLOM.pprint(0,'TSUITE y_test       =',trues.min(),trues.max(),trues.mean())
pyLOM.pprint(0,'TSUITE y_pred       =',pred_classes.min(),pred_classes.max(),pred_classes.mean())
pyLOM.pprint(0,'TSUITE mean_error   =',mean_error)

pyLOM.cr_info()
pyLOM.pprint(0,'End of output')
