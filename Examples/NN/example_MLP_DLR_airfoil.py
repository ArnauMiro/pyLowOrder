import numpy as np
import pyLOM
from pathlib import Path
import torch
import optuna


DATASET_DIR = Path("DLR_DATA")

input_scaler = pyLOM.NN.MinMaxScaler()
output_scaler = pyLOM.NN.MinMaxScaler()   

def load_dataset(path):
    original_dataset = pyLOM.Dataset.load(path)
    dataset = pyLOM.NN.Dataset(
        variables_out=(original_dataset["CP"],), 
        variables_in=original_dataset.xyz,
        parameters=[[*zip(original_dataset.get_variable('AoA'), original_dataset.get_variable('Mach'))]], # to have each Mach and AoA pair just once. To have all possibnle combinations, use [original_dataset.get_variable('AoA'), original_dataset.get_variable("Mach")]
        inputs_scaler=input_scaler,
        outputs_scaler=output_scaler,
    )
    return dataset

dataset_train = load_dataset(DATASET_DIR / "TRAIN.h5")
dataset_test = load_dataset(DATASET_DIR / "TEST.h5")
val_dataset = load_dataset(DATASET_DIR / "VAL.h5")

print(len(dataset_train), len(dataset_test), len(val_dataset))

x, y = dataset_train[:]
print(x.min(dim=0), x.max(dim=0), y.min(dim=0), y.max(dim=0), x.shape, y.shape)
print(x.shape, y.shape)

x, y = dataset_test[:]
print(x.min(), x.max(), y.min(), y.max(), x.shape, y.shape)

x, y = val_dataset[:]
print(x.min(), x.max(), y.min(), y.max(), x.shape, y.shape)

# define the optimizer
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
    p_dropouts=[0.15, 0.15],
)

training_params = {
    "epochs": 150,
    'lr_scheduler_step': 1,
    "optimizer_class": torch.optim.Adam,
    "loss_fn": torch.nn.MSELoss(),
    "print_rate_epoch": 1,
    "num_workers": 6,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "lr": 0.0008380427541690664, 
    "lr_gamma": 0.9905178804615045, 
    "batch_size": 119, 
    "hidden_size": 129, 
    "n_layers": 6
}

pipeline = pyLOM.NN.Pipeline(
    train_dataset=dataset_train,
    test_dataset=dataset_test,
    valid_dataset=val_dataset,
    # optimizer=optimizer,
    # model_class=pyLOM.NN.MLP,
    model=model,
    training_params=training_params,
)

pipeline.run()

# check saving and loading the model
pipeline.model.save("model.pth")
model = pyLOM.NN.MLP.load("model.pth")

preds = model.predict(dataset_test, batch_size=2048)
scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y = output_scaler.inverse_transform([dataset_test[:][1]])[0]

# check that the scaling is correct
print(scaled_y.min(), scaled_y.max())

print(f"MAE: {np.abs(scaled_preds - np.array(scaled_y)).mean()}")
print(f"MRE: {np.abs(scaled_preds - np.array(scaled_y)).mean() / abs(np.array(scaled_y).mean() + 1e-6)}")
print(f"MSE: {((scaled_preds - np.array(scaled_y)) ** 2).mean()}")
print(f"r2 score: {np.corrcoef(scaled_preds, np.array(scaled_y))[0, 1] ** 2}")