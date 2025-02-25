# Example of PINN on Burgers equation.
# The PDE is:
#   u_t + u * u_x - 0.01 * u_xx = 0, x in [-1, 1], t in [0, 1]
# with initial condition:
#   u(0, x) = -sin(pi * x), x in [-1, 1]
# and boundary conditions:
#   u(t, -1) = u(t, 1) = 0, t in [0, 1]


import numpy as np
import matplotlib.pyplot as plt
import torch
import pyLOM, pyLOM.NN


device = 'cpu'
RESUDIR = 'PINN_Burgers'
pyLOM.NN.create_results_folder(RESUDIR)

# Define the domain and the amount of points to sample
POINTS_ON_X = 256
POINTS_ON_T = 100
num_train_simulations = 5000
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
    n_layers=4,
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
    'epochs': 3000,
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
burgers_pinn.plot_training_logs(model_logs)
plt.savefig(RESUDIR + '/adam_train_test_loss.png')

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
burgers_pinn.plot_training_logs(model_logs)
plt.savefig(RESUDIR + '/lbfgs_train_test_loss.png')
pyLOM.pprint(0, f"Residual: {model_logs['test_loss'][-1]}")

print("Saving model")
burgers_pinn.save(RESUDIR + '/burgers_pinn.pt')
print("Model saved")
burgers_pinn_loaded = pyLOM.NN.BurgersPINN.load(RESUDIR + '/burgers_pinn.pt', device=device)

print("Model loaded")

# Predict and plot the results
u = burgers_pinn.predict(test_dataset).reshape(POINTS_ON_X, POINTS_ON_T)

plt.figure(figsize=(10, 5))
plt.imshow(u, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='coolwarm')
plt.title('Predicted')
plt.xlabel('t')
plt.ylabel('x')
plt.colorbar()
plt.savefig(RESUDIR + '/predicted.png')
plt.show()

num_time_snapshots = 3
x = torch.linspace(-1, 1, 256).reshape(-1, 1)
fig, axs = plt.subplots(1, num_time_snapshots, figsize=(15, 5))
for i in range(num_time_snapshots):
    instant = i / (num_time_snapshots - 1)
    t = torch.full_like(x, instant)
    u_instant_t = burgers_pinn(torch.cat([t, x], dim=-1).to(device)).detach().cpu().numpy().reshape(-1)

    axs[i].title.set_text(f"t = {instant}")
    axs[i].plot(x, u_instant_t, '.', label='Predicted')
    axs[i].legend()

plt.savefig(RESUDIR + '/time_snapshots.png')
plt.show()

pyLOM.cr_info()