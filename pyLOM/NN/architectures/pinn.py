from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
from torch.utils.data import DataLoader

from ... import pprint, cr # pyLOM/__init__.py
from ..utils import Dataset


class PINN(ABC):
    """
    This class represents a Physics-Informed Neural Network (PINN) model. It is an abstract class that needs to be subclassed to implement the pde_loss method.
    That method should compute the residual from the partial differential equation (PDE) and then compute the loss from it (usually by squaring the residual).

    Args:
        neural_net (torch.nn.Module): A neural network model that implements torch.nn.Module.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').

    Attributes:
        device (str): The device the model is running on.
        model (torch.nn.Module): The neural network model.

    """

    def __init__(self, neural_net, device):
        self.device = device
        self.model = neural_net.to(device)

    def __call__(self, x):
        """
        Forward pass of the PINN model.

        Args:
            x (torch.Tensor): The input tensor with the PDE input parameters.

        Returns:
            torch.Tensor: The output tensor, i.e. the solution for the PDE on x.

        """
        return self.model(x)

    def _prepare_input_variables(self, x_batch):
        """
        Prepares the input variables for training.

        Args:
            x_batch (torch.Tensor): The input batch tensor.

        Returns:
            List[torch.Tensor]: The list of prepared input variables.

        """
        input_variables = []
        
        for input_variable in range(x_batch.shape[1]):
            flow_variable = x_batch[:, input_variable : input_variable + 1]
            flow_variable.requires_grad_(True)
            input_variables.append(flow_variable)
        return input_variables

    def _get_dataloader(self, dataset, batch_size=None):
        """
        Creates data loaders for training.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor, optional): The target tensor. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to None.

        Returns:
            Union[torch.utils.data.DataLoader, List[torch.Tensor]]: The data loaders.

        """
        if batch_size is not None:
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)
        else:
            data_loader = [dataset[:]]
        return data_loader

    def bc_data_loss(self, pred, y, boundary_conditions, use_bfloat16=False):
        """
        Computes the loss from boundary conditions and data.

        Args:
            pred (torch.Tensor): The predicted output tensor.
            y (torch.Tensor): The target tensor.
            boundary_conditions (List[BoundaryCondition]): The list of boundary conditions.
            use_bfloat16 (bool, optional): Whether to use bfloat16 precision. Defaults to False.

        Returns:
            List[torch.Tensor]: The list of loss tensors.

        """
        if use_bfloat16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bc_losses = [bc.loss(self.model(bc.points.to(self.device))) for bc in boundary_conditions]
        else:
            bc_losses = [bc.loss(self.model(bc.points.to(self.device))) for bc in boundary_conditions]

        if y is not None:
            data_loss = torch.nn.functional.mse_loss(pred, y.to(self.device))
            bc_losses.append(data_loss)
        return bc_losses

    def compute_loss(self, x, y, boundary_conditions, use_bfloat16=False):
        """
        Computes the total loss for training.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The target tensor.
            boundary_conditions (List[BoundaryCondition]): The list of boundary conditions.
            use_bfloat16 (bool, optional): Whether to use bfloat16 precision. Defaults to False.

        Returns:
            List[torch.Tensor]: The list of loss tensors.

        """
        input_variables = self._prepare_input_variables(x)
        input_tensor = torch.cat(input_variables, dim=1).to(self.device)

        if use_bfloat16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = self.model(input_tensor)
        else:
            pred = self.model(input_tensor)

        return [self.pde_loss(pred, *input_variables)] + self.bc_data_loss(pred, y, boundary_conditions, use_bfloat16)

    @cr("PINN.fit")
    def fit(
        self,
        train_dataset: Dataset,
        optimizer_class=torch.optim.Adam,
        optimizer_params={},
        lr_scheduler_class=None,
        lr_scheduler_params={},
        epochs=1000,
        boundary_conditions=[],
        update_logs_steps=1,
        loaded_logs=None,
        batch_size=None,
        eval_dataset: Dataset = None,
        use_bfloat16=False,
        **kwargs,
    ):
        """
        Trains the PINN model.

        Args:
            train_dataset (Dataset): The training dataset. If the dataset returns a tuple, the first element is the input and the second element is the target. If not, the PINN is trained without simulation data.
            optimizer_class (torch.optim.Optimizer, optional): The optimizer class. Defaults to ``torch.optim.Adam``.
            optimizer_params (dict, optional): The optimizer parameters. Defaults to ``{}``.
            lr_scheduler_class (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler class. Defaults to ``None``.
            lr_scheduler_params (dict, optional): The learning rate scheduler parameters. Defaults to ``{}``.
            epochs (int, optional): The number of epochs to train for. Defaults to ``1000``.
            boundary_conditions (List[BoundaryCondition], optional): The list of boundary conditions. Defaults to ``[]``.
            update_logs_steps (int, optional): The interval for updating the progress. Defaults to ``100``.
            loaded_logs (dict, optional): Loaded training logs to be used as initial logs. Defaults to ``None``.
            batch_size (int, optional): The batch size. If none, the batch size will be equal to the number of collocation points given on `train_dataset`. Defaults to ``None``.
            eval_dataset (BaseDataset, optional): The evaluation dataset. Defaults to ``None``.
            use_bfloat16 (bool, optional): Whether to use bfloat16 precision. Defaults to ``False``.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The training logs.

        """
        logs = (
            loaded_logs
            if loaded_logs is not None
            else {
                "loss_from_pde": [],
                "loss_from_data_and_bc": [],
                "total_loss": [],
            }
        )

        train_data_loader = self._get_dataloader(train_dataset, batch_size)
        test_data_loader = None
        if eval_dataset is not None:
            test_data_loader = self._get_dataloader(eval_dataset, batch_size)
            if "test_loss" not in logs:
                logs["test_loss"] = []

        optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        if lr_scheduler_class is not None:
            lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_params)

        def closure():
            x_batch = batch[0].to(self.device)
            y_batch = batch[1].to(self.device) if len(batch) == 2 else None

            optimizer.zero_grad()
            losses = self.compute_loss(x_batch, y_batch, boundary_conditions, use_bfloat16)
            loss = sum(losses)
            loss.backward()
            
            loss_from_pde = losses[0].item()
            logs["loss_from_pde"].append(loss_from_pde)
            logs["loss_from_data_and_bc"].append(loss.item() - loss_from_pde)
            logs["total_loss"].append(loss.item())

            if update_logs_steps != 0 and (epoch % update_logs_steps == 0):
                extended_desc = ''
                if len(losses) > 1:
                    extended_desc = f", data/bc losses: [{', '.join(f'{x:.4e}' for x in losses[1:])}]"
                if 'test_loss' in logs and len(logs['test_loss']) > 0:
                    extended_desc += f", test loss: {logs['test_loss'][-1]:.4e}"
                desc = f"Epoch {epoch+1}/{epochs} Iteration {closure.iteration}. Pde loss: {loss_from_pde:.4e}" + extended_desc
                pprint(0, desc)

            closure.iteration += 1
            return loss

        self.model.train()

        for epoch in range(epochs):
            closure.iteration = 0

            for batch in train_data_loader: 
                optimizer.step(closure=closure)
                if lr_scheduler_class is not None:
                    lr_scheduler.step()

            if test_data_loader is not None:
                self.model.eval()
                test_loss = 0
                for batch in test_data_loader:
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device) if len(batch) == 2 else None
                    losses = self.compute_loss(x_batch, y_batch, boundary_conditions)
                    test_loss += sum(losses).item()
                logs["test_loss"].append(test_loss / len(test_data_loader))
                self.model.train()

        return logs
    
    @cr("PINN.predict")
    def predict(self, X: Dataset, **kwargs) -> np.ndarray:
        """
        Predicts for the input dataset.

        Args:
            X (Dataset): The input dataset.

        Returns:
            np.ndarray: The predictions of the model.

        """
        self.model.eval()
        data = X[:]
        input_data = data[0] # keep only the input data
        return self.model(input_data.to(self.device)).detach().cpu().numpy()

    def __repr__(self):
        """
        Returns a string representation of the PINN model.

        Returns:
            str: The string representation.

        """
        pprint(0, f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        return self.model.__repr__()

    def plot_training_logs(self, logs):
        """
        Plots the training logs.

        Args:
            logs (dict): The training logs.

        """
        plt.figure(figsize=(10, 6))
        plt.plot(logs["loss_from_pde"], label="PDE Loss")
        plt.plot(logs["loss_from_data_and_bc"], label="Data Conditions and BC Loss")
        plt.plot(logs["total_loss"], label="Total Loss")
        if "test_loss" in logs and len(logs["test_loss"]) != 0:
            total_epochs = len(logs["test_loss"]) 
            total_iters = len(logs["total_loss"]) 
            iters_per_epoch = total_iters // total_epochs
            plt.plot(np.arange(iters_per_epoch + total_iters % total_epochs, total_iters+1, step=iters_per_epoch), logs["test_loss"], label="Test Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training Losses")
        plt.legend()
        plt.show()

    @abstractmethod
    def pde_loss(self, pred, *input_variables):
        """
        Computes the loss from the partial differential equation (PDE).

        Args:
            pred (torch.Tensor): The predicted output tensor.
            *input_variables (torch.Tensor): The input variables for the PDE. e.g. x, y, t.

        Returns:
            torch.Tensor: The loss tensor.

        """
        pass

    def save(self, path):
        """
        Saves the model to a file using torchscript.

        Args:
            path (str): The path to save the model.

        """

        path = Path(path)
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(path)
        

    @classmethod
    def load(cls, path, device='cpu'):
        """
        Loads the model from a file.

        Args:
            path (str): The path to load the model.
            neural_net (torch.nn.Module): The neural network model.
            device (str, optional): The device to run the model on. Defaults to 'cpu'.

        Returns:
            PINN: The loaded PINN model.

        """
        model = torch.jit.load(path, map_location=device)
        return cls(neural_net=model, device=device)

class BoundaryCondition(ABC):
    """
    Abstract base class for defining boundary conditions. You need to implement the `loss` method to use a custom boundary condition.

    Args:
        points (Tensor): The points where the boundary condition is defined.

    Attributes:
        points (Tensor): The points where the boundary condition is defined.
    """
    def __init__(self, points):
        self._points = points
        self._points.requires_grad_(True)
    
    @abstractmethod
    def loss(self, pred):
        """
        Computes the loss for the given prediction.

        Args:
            pred (Tensor): The predicted values on the points where the boundary condition is defined.

        Returns:
            Tensor: The loss value.
        """
        pass

    @property
    def points(self):
        return self._points


class DirichletCondition(BoundaryCondition):
    """
    This class represents a Dirichlet boundary condition.

    Args:
        points (Tensor): The predicted values on the points where the boundary condition is defined.
        values (Tensor): The values of the boundary condition.
    """
    def __init__(self, points, values):
        super().__init__(points)
        self._values = values

    def loss(self, pred):
        return torch.mean((pred - self._values) ** 2)

class BurgersPINN(PINN):
    r"""
    This class represents a Physics-Informed Neural Network (PINN) model for the Burgers' equation.
    The model predictions have 1 column, the velocity field :math:`u`.

    .. math::
        \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2u}{\partial x^2}

    Args:
        neural_net (torch.nn.Module): The neural network model.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        viscosity (float): The viscosity coefficient.
    """
    def __init__(self, neural_net, device, viscosity=0.01):
        super().__init__(neural_net, device)
        self.viscosity = viscosity

    def pde_loss(self, pred, *input_variables):
        t, x = input_variables
        u = pred
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        f = u_t + u * u_x - (self.viscosity / torch.pi) * u_xx

        return (f ** 2).mean()

class NavierStokesIncompressible2DPINN(PINN):
    r"""
    This class represents a Physics-Informed Neural Network (PINN) model for the incompressible steady 2D Navier-Stokes equations.
    The model predictions have 3 columns, the velocity field :math:`(u, v)` and the pressure :math:`p` fields.
    
    .. math::
        \begin{align*}
        u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \frac{1}{Re} \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) &= 0, \\
        u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \frac{1}{Re} \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) &= 0, \\
        \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} &= 0.
        \end{align*}

    Args:
        neural_net (torch.nn.Module): The neural network model.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        Re (float): The Reynolds number.
    """

    def __init__(self, neural_net, device, Re=50):
        super().__init__(neural_net, device)
        self.Re = Re
    
    def pde_loss(self, pred , x, y):

        u, v, p= pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]        

        # Spatial gradients
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        # Second order spatial gradients
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        
        # Continuity equation
        continuity = u_x + v_y
        
        # Momentum equations
        momentum_u = u * u_x + v * u_y + p_x - (1 / self.Re) * (u_xx + u_yy)
        momentum_v = u * v_x + v * v_y + p_y - (1 / self.Re) * (v_xx + v_yy)

        # Total loss
        loss_continuity = torch.mean(continuity ** 2)
        loss_momentum_u = torch.mean(momentum_u ** 2)
        loss_momentum_v = torch.mean(momentum_v ** 2)
        
        loss = loss_continuity + loss_momentum_u + loss_momentum_v
        return loss

class Euler2DPINN(PINN):
    # TODO: review the equations once sphinx documentation is working
    r"""
    This class represents a Physics-Informed Neural Network (PINN) model for the 2D Euler equations.
    The model predictions have 4 columns, the density :math:`\rho`, the velocity field :math:`(u, v)` and the total energy :math:`E` fields.

    .. math::
        \begin{align*}
        \frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} + \frac{\partial (\rho v)}{\partial y} &= 0, \\
        \frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial x} + \frac{\partial (\rho uv)}{\partial y} &= 0, \\
        \frac{\partial (\rho v)}{\partial t} + \frac{\partial (\rho uv)}{\partial x} + \frac{\partial (\rho v^2 + p)}{\partial y} &= 0, \\
        \frac{\partial (\rho E)}{\partial t} + \frac{\partial (u(\rho E + p))}{\partial x} + \frac{\partial (v(\rho E + p))}{\partial y} &= 0.
        \end{align*}

    Args:
        neural_net (torch.nn.Module): The neural network model.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
    """

    GAMMA = 1.4

    def pde_loss(self, pred, *input_variables):

        x_coord = input_variables[0]
        y_coord = input_variables[1]
        rho, u, v, E = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]

        p = (Euler2DPINN.GAMMA - 1.0) * (E - 0.5 * rho * (u**2 + v**2))

        # F
        F1 = rho * u
        F2 = rho * u**2 + p
        F3 = rho * u * v
        F4 = u * (E + p)

        # G
        G1 = rho * v
        G2 = rho * u * v
        G3 = rho * v**2 + p
        G4 = v * (E + p)

        dF1_dx = torch.autograd.grad(F1, x_coord, grad_outputs=torch.ones_like(F1), create_graph=True)[0]
        dF2_dx = torch.autograd.grad(F2, x_coord, grad_outputs=torch.ones_like(F2), create_graph=True)[0]
        dF3_dx = torch.autograd.grad(F3, x_coord, grad_outputs=torch.ones_like(F3), create_graph=True)[0]
        dF4_dx = torch.autograd.grad(F4, x_coord, grad_outputs=torch.ones_like(F4), create_graph=True)[0]

        dG1_dy = torch.autograd.grad(G1, y_coord, grad_outputs=torch.ones_like(G1), create_graph=True)[0]
        dG2_dy = torch.autograd.grad(G2, y_coord, grad_outputs=torch.ones_like(G2), create_graph=True)[0]
        dG3_dy = torch.autograd.grad(G3, y_coord, grad_outputs=torch.ones_like(G3), create_graph=True)[0]
        dG4_dy = torch.autograd.grad(G4, y_coord, grad_outputs=torch.ones_like(G4), create_graph=True)[0]

        residual_mass_conserv = dF1_dx + dG1_dy
        residual_momentum_x = dF2_dx + dG2_dy
        residual_momentum_y = dF3_dx + dG3_dy
        residual_energy = dF4_dx + dG4_dy

        loss_mass_conserv = torch.mean((residual_mass_conserv) ** 2) 
        loss_momentum_x = torch.mean((residual_momentum_x) ** 2)  
        loss_momentum_y = torch.mean((residual_momentum_y) ** 2)  
        loss_energy = torch.mean((residual_energy) ** 2) 

        loss = loss_mass_conserv + loss_momentum_x + loss_momentum_y + loss_energy
        return loss