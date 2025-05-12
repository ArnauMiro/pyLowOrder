#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN interpolation routines.
#
# Last rev: 12/05/2025

import torch

from ..dataset        import Dataset as pyLOMDataset

class Interpolator():
    def __init__(
        self, 
        dataset: pyLOMDataset,
    ):
        self.dataset = dataset

    def adjust_field(
        self,
        fieldname,
        obj_func,
        get_opt_param_func,
        constr_func=None,
        optimizer_class=torch.optim.Adam, 
        schduler_class=torch.optim.lr_scheduler.StepLR,
        opt_config=None,
        disp_progress=(False, 0),
        **kwargs
    ):
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
                print(f"\nCase {i}:")

            prev_loss = float('inf')
            n_improvement = 0

            for epoch in range(config['niter']):
                total_loss, obj_loss, penalty = optimizer.step(lambda: closure(colTensor, colTensor0, opt_vars))
                scheduler.step()
                losses.append([obj_loss.item(), penalty.item(), total_loss.item()])

                if disp_progress[0] and (epoch % disp_progress[1] == 0):
                    print(f"Epoch {epoch:4}: Total Loss = {total_loss.item():.2e}, Objective = {obj_loss.item():.2e}, Penalty = {penalty.item():.2e}")

                loss_diff = abs(prev_loss - total_loss.item())
                prev_loss = total_loss.item()
                if loss_diff < config['tolerance']:
                    n_improvement += 1
                else:
                    n_improvement = 0
                if n_improvement >= config['patience']:
                    print(f"Early stopping at epoch {epoch}, no significant improvement.")
                    break
                if epoch >= config['niter'] - 1:
                    print(f"Reached maximum number of epochs ({config['niter']}). Stopping.")
                    break

            field_mod[:, i] = colTensor.detach().numpy()
            field_losses.append(losses)

        ndim = self.dataset.info(fieldname)['ndim']
        self.dataset.add_field(varname=fieldname + 'Adjusted', ndim=ndim, var=field_mod)
        return self.dataset, field_losses