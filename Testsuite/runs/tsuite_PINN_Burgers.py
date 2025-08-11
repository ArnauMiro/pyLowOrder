# Example of PINN on Burgers equation.
# The PDE is:
#   u_t + u * u_x - 0.01 * u_xx = 0, x in [-1, 1], t in [0, 1]
# with initial condition:
#   u(0, x) = -sin(pi * x), x in [-1, 1]
# and boundary conditions:
#   u(t, -1) = u(t, 1) = 0, t in [0, 1]


import sys, os, numpy as np
import torch
import pyLOM, pyLOM.NN


DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]

device  = 'cpu'
RESUDIR = os.path.join(OUTDIR,f'PINN_{DATAFILE}')
pyLOM.NN.create_results_folder(RESUDIR,verbose=False)

# Define the domain and the amount of points to sample
POINTS_ON_X = 256
POINTS_ON_T = 100
num_train_simulations = 4000
t = np.linspace(0, 1, POINTS_ON_T)
x = np.linspace(-1, 1, POINTS_ON_X)
T, X = np.meshgrid(t, x)

idx = np.random.choice(X.flatten().shape[0], num_train_simulations, replace=False)
TX = np.concatenate([T.reshape(-1, 1), X.reshape(-1, 1)], axis=1)
TX = torch.tensor(TX).float()
train_TX = TX[idx]

# Define the boundary conditions
class InitialCondition(pyLOM.NN.BoundaryCondition):

    def loss(self, pred):
        x = self.points[:, 1].reshape(-1, 1)
        initial_cond_pred = pred
        # sin is positive here because the initial condition is u(0, x) = -sin(pi * x)
        ic_loss = (initial_cond_pred + torch.sin(torch.pi * x).to(device)) ** 2
        return ic_loss.mean()


class XBoudaryCondition(pyLOM.NN.BoundaryCondition):

    def loss(self, pred):
        # as u on the boundary is 0, we can just return the mean of the prediction
        return pred.pow(2).mean()


initial_points = torch.tensor(x).reshape(-1, 1)
initial_bc = InitialCondition(
    torch.cat([torch.full_like(initial_points, 0), initial_points], dim=-1).float(),
)

boundary_points = torch.tensor(t).reshape(-1, 1)
boundary_bc = XBoudaryCondition(
    torch.cat(
        [torch.cat([boundary_points, torch.full_like(boundary_points, -1)], dim=-1),
        torch.cat([boundary_points, torch.full_like(boundary_points, 1)], dim=-1),]
    ).float()
)

train_dataset = torch.utils.data.TensorDataset(train_TX)
test_dataset = torch.utils.data.TensorDataset(TX)

# Define the neural network
input_dim = TX.shape[1]
output_dim = 1 # u(t, x)

# The pinn needs a neural network, which can be any pytorch model that implements nn.Module
net = pyLOM.NN.MLP(
    input_size=input_dim,
    output_size=output_dim,
    hidden_size=40,
    n_layers=3,
    activation=torch.nn.functional.tanh, # With relu the model struggles to converge
)

burgers_pinn = pyLOM.NN.BurgersPINN(
    viscosity=0.01,
    neural_net=net,
    device=device,
)
pyLOM.pprint(0, burgers_pinn)

training_params = {
    'optimizer_class': torch.optim.Adam,
    'optimizer_params': {'lr': 1e-3},
    'lr_scheduler_class': torch.optim.lr_scheduler.StepLR,
    'lr_scheduler_params': {'step_size': 1000, 'gamma': 0.99},
    'epochs': 100,
    'update_logs_steps': 100,
    'boundary_conditions': [initial_bc, boundary_bc],
}

# Create the pipeline and train the model with Adam
pipeline_adam = pyLOM.NN.Pipeline(
    model=burgers_pinn,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    training_params=training_params
)
model_logs = pipeline_adam.run()



# Train the model with L-BFGS to improve the results
lbfgs_params = {
    'lr': 0.01,
    'max_iter': 12000,
    'max_eval': 10000,
    'history_size': 200,
    'tolerance_grad': 1e-12,
    'tolerance_change': 0.5 * np.finfo(float).eps,
    'line_search_fn': 'strong_wolfe'
}
training_params = {
    'optimizer_class': torch.optim.LBFGS,
    'optimizer_params': lbfgs_params,
    'loaded_logs': model_logs,
    'epochs': 1,
    'boundary_conditions': [initial_bc, boundary_bc],
}

pipeline_lbfgs = pyLOM.NN.Pipeline(
    model=burgers_pinn,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    training_params=training_params
)
model_logs = pipeline_lbfgs.run()
pyLOM.pprint(0, f"Residual: {model_logs['test_loss'][-1]}")

burgers_pinn.save(RESUDIR + '/burgers_pinn.pt')
pyLOM.pprint(0, "Model saved")
burgers_pinn_loaded = pyLOM.NN.BurgersPINN.load(RESUDIR + '/burgers_pinn.pt', device=device)

pyLOM.pprint(0, "Model loaded")

u = burgers_pinn.predict(test_dataset).reshape(POINTS_ON_X, POINTS_ON_T)

## Testsuite output
pyLOM.pprint(0,'TSUITE u     =',u.min(),u.max(),u.mean())

pyLOM.cr_info()
pyLOM.pprint(0,'End of output')