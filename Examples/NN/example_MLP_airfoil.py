from itertools import product
from typing import List

import numpy as np
import torch
import torch.utils
import torch.utils.data
from sklearn.preprocessing import StandardScaler

import pyLOM

# CSV_PATH = '/home/david/Desktop/CETACEO_UPM/use_cases/aerodynamics/pinn/data/casos_euler/dataset_csv'

# df = pd.read_csv(CSV_PATH + '/metadata.csv')
# print(df.dtypes)
# df = df.drop(columns=['vtu name'])
# x = df[['l.v.1', 'l.v.2', 'l.v.3', 'l.v.4']].values
# y = df[['CL', 'CD']].values

# train_idx = np.random.choice(range(x.shape[0]), int(0.8*x.shape[0]), replace=False)

# x_train, y_train = x[train_idx], y[train_idx]
# x_test, y_test = np.delete(x, train_idx, axis=0), np.delete(y, train_idx, axis=0)

# dataset_train = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
# dataset_test = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))



DATASET_PATH = "/home/david/Downloads/mean_yaw_dataset.h5"
dataset = pyLOM.Dataset.load(DATASET_PATH)
print(type(dataset["Cp"]), dataset["Cp"].shape, dataset.mesh.xyz.shape)

alphas = [2.5, 5, 7.5, 10]

train_idx = np.random.choice(
    range(dataset.mesh.xyz.shape[0]),
    int(0.8 * dataset.mesh.xyz.shape[0]),
    replace=False,
)

u = torch.rand(2000, 100)
v = torch.rand(2000, 100)

mesh_shape = (40, 10, 5)

# dataset = DatasetNuevo(variables_out=(u, v), mesh_shape=mesh_shape)

# print(len(dataset))
# x= dataset[:2]
# print(x.shape)
# x= dataset[:]
# print(x.shape)
# import sys; sys.exit(0)

y_train = dataset["Cp"][train_idx]
x_train = dataset.mesh.xyz[train_idx]
y_test = np.delete(dataset["Cp"], train_idx, axis=0)
x_test = np.delete(dataset.mesh.xyz, train_idx, axis=0)

dataset_train = pyLOM.NN.Dataset(
    variables_out=(y_train,), variables_in=x_train, parameters=[alphas[:-1]]
)
dataset_test = pyLOM.NN.DatasetNuevo(
    variables_out=(y_test,), variables_in=x_test, parameters=[alphas[-1]]
)

print(len(dataset_train))
x, y = dataset_train[0]
print(x.shape, y.shape)
x, y = dataset_train[:]
print(x.shape, y.shape)

# import sys; sys.exit(0)


model = pyLOM.NN.MLP(
    input_size=x.shape[1],
    output_size=y.shape[1],
    hidden_size=512,
    n_layers=3,
    p_dropouts=[0.07158, 0.03035, 0.15853],
)


optimizer = pyLOM.NN.OptunaOptimizer(
    optimization_params={
        "lr": 0.01,  # fixed parameter
        "n_layers": (1, 4),  # optimizable parameter,
        "batch_size": (128, 512),
        "hidden_size": 256,
        # 'p_dropouts': [0.07158, 0.03035, 0.15853],
        "epochs": 30,
    },
    n_trials=10,
    direction="minimize",
    pruner=None,
    save_dir=None,
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=dataset_train,
    test_dataset=dataset_test,
    # optimizer=optimizer,
    # model_class=pyLOM.NN.MLP,
    model=model,
    training_params={
        "batch_size": 2048,
        "epochs": 15,
        "lr": 0.01,
        "print_rate_epoch": 1,
    },
)
pipeline.run()
