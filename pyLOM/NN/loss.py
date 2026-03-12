#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN loss functions.
#
# Last rev: 12/03/2026

import torch 

from abc            import ABC, abstractmethod

class BaseLossFunction(torch.nn.Module, ABC):
    r"""
    Base abstract class for loss functions. 
    All loss functions must inherit from this class and implement the forward method, which computes the loss given a model and a batch of data.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, model, batch):
        pass

class TorchLossAdapter(BaseLossFunction):
    r"""
    Adapter to wrap standard PyTorch loss functions into the BaseLossFunction interface.
    
    Args:
        torch_loss: An instance of a PyTorch loss function (e.g., torch.nn.MSELoss()).
    """
    def __init__(self, torch_loss):
        super().__init__()
        self.torch_loss = torch_loss

    def forward(self, model, batch):
        x, y = batch["x"], batch["y"]
        pred = model(x)
        return self.torch_loss(pred, y)
    
class GradientWeightedMSE(BaseLossFunction):
    r"""
    Gradient-weighted Mean Squared Error loss function.
    The loss computes the standard MSE between predictions and targets, but weights it by the magnitude of the gradient of the predictions with respect to the input features. 
    This gives more importance to points where the model output changes rapidly with respect to the input, which can help in learning sharper features in the data.

    Args:
        alpha: Weighting factor for the gradient term. Higher values give more importance to the gradient weighting (default: ``1.0``).
        geom_dim: Number of geometric input dimensions to consider for the gradient (e.g., 3 for 3D spatial coordinates; default: ``3``).
        eps: Small constant to prevent division by zero when normalizing the gradient magnitude (default: ``1e-8``).
    """
    def __init__(
        self,
        alpha: float = 1.0,
        geom_dim: int = 3,
        eps: float = 1e-8
    ):
        super().__init__()
        self.alpha = alpha
        self.geom_dim = geom_dim
        self.eps = eps

    def forward(
        self,
        model,
        batch
    ) -> torch.Tensor:

        x, y = batch["x"], batch["y"]
        
        # Predict outputs with gradient tracking enabled
        x = x.clone().detach().requires_grad_(True)
        pred = model(x)

        # Pointwise MSE
        mse_pointwise = (pred - y) ** 2

        # Gradient magnitude with respect to geometric inputs
        grads_full = torch.autograd.grad(
            outputs=pred,
            inputs=x,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        grads_geom = grads_full[:, 0:self.geom_dim]
        grad_norm = torch.norm(grads_geom, dim=1)
        grad_weight = grad_norm / (grad_norm.mean() + self.eps)

        # Weighted MSE
        weighted_mse = (1.0 + self.alpha * grad_weight.unsqueeze(-1)) * mse_pointwise
        return weighted_mse.mean()

