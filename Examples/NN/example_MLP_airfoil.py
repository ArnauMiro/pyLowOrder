from itertools import product
from typing import List

import numpy as np
import torch
import torch.utils
import torch.utils.data

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

class DatasetNuevo(torch.utils.data.Dataset):
    def __init__(self, variables_out: tuple, mesh_shape:tuple = (1,), variables_in: np.ndarray = None, parameters: List[List[float]] = None):
        self.parameters = parameters
        self.num_channels = len(variables_out)
        self.mesh_shape = mesh_shape
        self.variables_out = self._process_variables_out(variables_out)
        self.variables_in = self._process_variables_in(variables_in, parameters) if variables_in is not None else None

    def _process_variables_out(self, variables_out):
        if len(variables_out) == 1:
            variables_out = torch.tensor(variables_out[0])
        else:
            variables_out = torch.cat([torch.tensor(variable).unsqueeze(0) for variable in variables_out], dim=0) # (C, mul(mesh_shape), N)
        if self.num_channels == 1:
            variables_out = variables_out.unsqueeze(-1)
        print(variables_out.shape)
        variables_out = variables_out.permute(2, 0, 1) # (N, C, mul(mesh_shape))
        print(variables_out.shape)
        variables_out = variables_out.reshape(-1, self.num_channels, *self.mesh_shape) # (N, C, *mesh_shape)
        print(variables_out.shape)
        if variables_out.shape[-1] == 1: #(N, C, 1) -> (N, C)
            variables_out = variables_out.squeeze(-1)
        return variables_out.float()
    
    def _process_variables_in(self, variables_in, parameters):
        if parameters is None:
            variables_in = torch.tensor(variables_in, dtype=torch.float32)
            return variables_in
        variables_in = torch.tensor(variables_in, dtype=torch.float32)
        # parameters is a list of lists of floats. Each contains the values that will be repeated for each input coordinate
        # in some sense, it is like a cartesian product of the parameters with the input coordinates
        cartesian_product = list(product(*parameters))
        cartesian_product = torch.tensor(cartesian_product)
        variables_in_repeated = variables_in.repeat(len(cartesian_product), 1)
        cartesian_product = cartesian_product.repeat(len(variables_in), 1)
        return torch.cat([variables_in_repeated, cartesian_product], dim=1).float()

    def __len__(self):
        return len(self.variables_out)

    def __getitem__(self, idx):
        if self.variables_in is None:
            return self.variables_out[idx]
        return self.variables_in[idx], self.variables_out[idx]

DATASET_PATH = '/home/david/Downloads/mean_yaw_dataset.h5'
dataset = pyLOM.Dataset.load(DATASET_PATH)
print(type(dataset['Cp']), dataset['Cp'].shape, dataset.mesh.xyz.shape)

alphas = [2.5, 5, 7.5, 10]

train_idx = np.random.choice(range(dataset.mesh.xyz.shape[0]), int(0.8*dataset.mesh.xyz.shape[0]), replace=False)

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

y_train = dataset['Cp'][train_idx]
x_train = dataset.mesh.xyz[train_idx]
y_test = np.delete(dataset['Cp'], train_idx, axis=0)
x_test = np.delete(dataset.mesh.xyz, train_idx, axis=0)

dataset_train = DatasetNuevo(variables_out=(y_train,), variables_in=x_train, parameters=[alphas[:-1]])
dataset_test = DatasetNuevo(variables_out=(y_test,), variables_in=x_test, parameters=[alphas[-1]])

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
        p_dropouts=[0.07158, 0.03035, 0.15853]
    )


optimizer = pyLOM.NN.OptunaOptimizer(
    optimization_params={
        "lr": 0.01,  # fixed parameter
        "n_layers": (1, 4),  # optimizable parameter,
        'batch_size': (128, 512),
        'hidden_size': 256,
        # 'p_dropouts': [0.07158, 0.03035, 0.15853],
        'epochs': 30,
    },
    n_trials=10,
    direction='minimize',
    pruner=None,
    save_dir=None
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=dataset_train,
    test_dataset=dataset_test,
    # optimizer=optimizer,
    # model_class=pyLOM.NN.MLP,
    model=model,
    training_params={
        'batch_size': 2048,
        'epochs': 15,
        'lr': 0.01,
        'print_rate_epoch': 1,
    },
)
pipeline.run()