#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN interpolation routines.
#
# Last rev: 22/05/2025

import torch

from ..dataset          import Dataset as pyLOMDataset
from ..utils.errors     import raiseError
from ..                 import pprint

class Interpolator():
    def __init__(
        self, 
        dataset: pyLOMDataset,
    ):
        self.dataset = dataset

    @staticmethod
    def objective_mse(
        field_mod: torch.Tensor, 
        field_ref: torch.Tensor, 
        **kwargs: dict,
    )-> torch.Tensor:
        r"""
        Objective function to minimize the difference between the modified and original field.

        Args:
            field_mod (torch.Tensor): Modified field.
            field_ref (torch.Tensor): Original field.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The sum of squared differences between the modified and original field.
        """
        mse_loss = torch.nn.MSELoss(reduction='sum')
        return mse_loss(field_mod, field_ref)

    @staticmethod
    def multitarget_equality_penalty(
        field_mod: torch.Tensor,
        penalty_func: callable,
        target_names: list,
        ref_values: dict, 
        penalty_args: dict, 
        **kwargs,
    )-> torch.Tensor:
        r"""
        Multitarget equality penalty function to ensure the modified field matches the reference values.
        
        Args:
            field_mod (torch.Tensor): Modified field.
            penalty_func (callable): Function to compute the penalty.
            target_names (list): List of target names.
            ref_values (dict): Dictionary with reference values for each target.
            penalty_args (dict): Additional arguments for the penalty function.
            **kwargs: Additional arguments.
        
        Returns:
            torch.Tensor: The sum of squared differences between the modified field and the reference values.
        """
        targets = penalty_func(field_mod, **penalty_args)
        penalty = 0.0
        ref0 = ref_values[target_names[0]]
        for i, name in enumerate(target_names):
            diff = (targets[i] - ref_values[name])**2
            factor = ref0/ref_values[name]
            penalty += diff * factor
        return penalty

    @staticmethod
    def get_opt_params_for_case(
        dataset: pyLOMDataset,
        i: int,
        **kwargs,
    )-> dict:
        r"""
        Get optimization parameters for a specific case.

        Args:
            dataset (pyLOMDataset): The dataset containing the fields.
            i (int): Index of the current case.
            **kwargs: Additional arguments.
        
        Returns:
            dict: A dictionary containing the optimization parameters.
        """
        mapping = kwargs.get('opt_param_config', {})

        ref_values = {}
        penalty_args = {}

        for original_name, (source_type, func_arg_name) in mapping.items():
            if source_type == 'get_variable':
                val = dataset.get_variable(original_name)[i]
            elif source_type == 'field':
                val = dataset[original_name]
            else:
                raiseError(f"Unknown source type '{source_type}' for variable '{original_name}'")

            if func_arg_name in kwargs.get('target_names', []):
                ref_values[func_arg_name] = torch.tensor(val)
            else:
                penalty_args[func_arg_name] = torch.tensor(val)

        return {
            'ref_values': ref_values,
            'penalty_args': penalty_args,
            'penalty_func': kwargs.get('penalty_func'),
            'target_names': kwargs.get('target_names'),
        }

    def adjust_field(
        self,
        fieldname: str,
        obj_func: callable = objective_mse,
        get_opt_param_func: callable = get_opt_params_for_case,
        constr_func: callable = multitarget_equality_penalty,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        schduler_class: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR,
        opt_config: dict = None,
        disp_progress: tuple = (False, 0),
        **kwargs
    )-> tuple[pyLOMDataset, list]:
        r"""
        Adjusts a field in the dataset using an optimization algorithm.

        Args:
            fieldname (str): Name of the field to be adjusted.
            obj_func (callable): Objective function to minimize.
            get_opt_param_func (callable): Function to get optimization parameters.
            constr_func (callable, optional): Constraint function to apply (default: ``None``).
            optimizer_class (torch.optim.Optimizer, optional): Optimizer class to use (default: ``torch.optim.Adam``).
            schduler_class (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler class (default: ``torch.optim.lr_scheduler.StepLR``).
            opt_config (dict, optional): Configuration for the optimizer (default: ``None``).
                - niter (int): Number of iterations (default: ``1000``).
                - lr (float): Learning rate (default: ``1e-2``).
                - lr_step_size (int): Step size for the learning rate scheduler (default: ``1``).
                - lr_gamma (float): Gamma for the learning rate scheduler (default: ``0.999``).
                - penalty_factor (float): Penalty factor for the constraint function (default: ``1e5``).
                - tolerance (float): Tolerance for early stopping (default: ``1e-9``).
                - patience (int): Number of iterations with no improvement before stopping (default: ``10``).
            disp_progress (tuple, optional): Tuple containing a boolean for displaying progress and an integer for the display frequency (default: ``(False, 0)``).
            **kwargs: Additional arguments for the objective and constraint functions.

        Returns:
            tuple: A tuple containing the modified dataset and a list of losses for each case.
        """
        default_config = {
            'niter': 1000,
            'lr': 1e-2,
            'lr_step_size': 1,
            'lr_gamma': 0.999,
            'penalty_factor': 1e5,
            'tolerance': 1e-9,
            'patience': 10,
        }

        if opt_config is None:
            opt_config = {}
        config = {**default_config, **opt_config}

        field = self.dataset[fieldname]
        field_mod = field.copy()
        field_losses = []

        def closure(colTensor, colTensor0, opt_vars):
            optimizer.zero_grad()
            obj_loss = obj_func(colTensor, colTensor0, **opt_vars)
            if constr_func is not None:
                penalty = constr_func(colTensor, **opt_vars)
                total_loss = obj_loss + config['penalty_factor'] * penalty
                total_loss.backward()
                return total_loss, obj_loss, penalty
            else:
                total_loss = obj_loss
                total_loss.backward()
                return total_loss, obj_loss, torch.tensor(0.0)

        for i, col in enumerate(field.T):
            colTensor = torch.tensor(col, requires_grad=True)
            colTensor0 = colTensor.clone().detach()
            opt_vars = get_opt_param_func(self.dataset, i, **kwargs)

            optimizer = optimizer_class([colTensor], lr=config['lr'])
            scheduler = schduler_class(optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])
            losses = []

            if disp_progress[0]:
                pprint(0, f"\nCase {i}:")

            prev_loss = float('inf')
            n_improvement = 0

            for epoch in range(config['niter']):
                total_loss, obj_loss, penalty = optimizer.step(lambda: closure(colTensor, colTensor0, opt_vars))
                scheduler.step()
                losses.append([obj_loss.item(), penalty.item(), total_loss.item()])

                if disp_progress[0] and (epoch % disp_progress[1] == 0):
                    pprint(0, f"Epoch {epoch:4}: Total Loss = {total_loss.item():.2e}, Objective = {obj_loss.item():.2e}, Penalty = {penalty.item():.2e}")

                loss_diff = abs(prev_loss - total_loss.item())
                prev_loss = total_loss.item()
                if loss_diff < config['tolerance']:
                    n_improvement += 1
                else:
                    n_improvement = 0
                if n_improvement >= config['patience']:
                    pprint(0, f"Early stopping at epoch {epoch}, no significant improvement.")
                    break
                if epoch >= config['niter'] - 1:
                    pprint(0, f"Reached maximum number of epochs ({config['niter']}). Stopping.")
                    break

            field_mod[:, i] = colTensor.detach().numpy()
            field_losses.append(losses)

        ndim = self.dataset.info(fieldname)['ndim']
        self.dataset.add_field(varname=fieldname + 'Adjusted', ndim=ndim, var=field_mod)
        return self.dataset, field_losses