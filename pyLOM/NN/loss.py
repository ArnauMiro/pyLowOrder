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
        per_output: If ``False`` (default), a single shared gradient weight is computed per point by differentiating the sum of all outputs. If ``True``, an independent gradient weight is computed for each output variable separately.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        geom_dim: int = 3,
        eps: float = 1e-8,
        per_output: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.geom_dim = geom_dim
        self.eps = eps
        self.per_output = per_output

    def _shared_grad_weight(
        self, 
        pred, 
        x
    ) -> torch.Tensor:
        r"""
        Compute scalar weight per point, shared across all outputs.
        Differentiates the sum of all outputs w.r.t. x, so the gradient reflects the aggregate spatial sensitivity of the model.
        """
        grads_full = torch.autograd.grad(
            outputs=pred,
            inputs=x,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]                                                        # [N, input_dim]
        grads_geom = grads_full[:, 0:self.geom_dim]                 # [N, geom_dim]
        grad_norm = torch.norm(grads_geom, dim=1)                   # [N]
        grad_weight = grad_norm / (grad_norm.mean() + self.eps)     # [N]
        return grad_weight.unsqueeze(-1)                            # [N, 1]

    def _per_output_grad_weight(
        self, 
        pred, 
        x
    ) -> torch.Tensor:
        r"""
        Compute one scalar weight per (point, output) pair.
        Differentiates each output independently, capturing output-specific spatial sensitivity.
        """
        grad_weights = []
        for j in range(pred.shape[1]):
            grad_j = torch.autograd.grad(
                outputs=pred[:, j].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
            )[0]                                                        # [N, input_dim]
            geom_grad_j = grad_j[:, 0:self.geom_dim]                    # [N, geom_dim]
            norm_j = torch.norm(geom_grad_j, dim=1)                     # [N]
            grad_weights.append(norm_j / (norm_j.mean() + self.eps))    # [N]
        return torch.stack(grad_weights, dim=1)                         # [N, M]

    def forward(
        self, 
        model, 
        batch
    ) -> torch.Tensor:
        
        x, y = batch["x"], batch["y"]

        # Predict outputs with gradient tracking enabled
        x = x.clone().detach().requires_grad_(True)
        pred = model(x)

        # Pointwise squared error [N, M]
        mse_pointwise = (pred - y) ** 2

        # Gradient-based weights [N, 1] or [N, M]
        if self.per_output:
            grad_weight = self._per_output_grad_weight(pred, x)
        else:
            grad_weight = self._shared_grad_weight(pred, x)

        # Weighted MSE
        weighted_mse = (1.0 + self.alpha * grad_weight) * mse_pointwise
        return weighted_mse.mean()

class NeighborDifferenceMSELoss(BaseLossFunction):
    r"""
    Neighbor Difference Mean Squared Error loss function.
    The loss computes the standard MSE between predictions and targets, but also considers the differences between neighboring points.
    This encourages the model to not only fit the target values at each point but also to match the differences between neighboring points, which can help in learning smoother and more physically consistent models.

    Args:
        alpha: Weighting factor for the neighbor difference term. Higher values give more importance to matching the differences between neighbors (default: ``1.0``).
        eps: Small constant to prevent division by zero when normalizing the neighbor differences (default: ``1e-8``).
    """
    def __init__(
        self,
        alpha: float = 1.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        model: torch.nn.Module,
        batch
    ) -> torch.Tensor:

        center_x, neighbor_x, center_y, neighbor_y = batch["x"], batch["x_neighbors"], batch["y"], batch["y_neighbors"]
        B, K, input_dim = neighbor_x.shape

        # Predict center and neighbor outputs
        pred_center = model(center_x)

        pred_neighbors = model(neighbor_x.view(-1, input_dim))
        pred_neighbors = pred_neighbors.view(B, K, -1)

        # Standard MSE on center points
        mse_loss = torch.mean((pred_center - center_y) ** 2)

        # Neighbor differences
        diff_true = neighbor_y - center_y.unsqueeze(1)
        diff_pred = pred_neighbors - pred_center.unsqueeze(1)

        diff_weight = torch.norm(diff_true, dim=-1)
        diff_weight = diff_weight / (diff_weight.mean() + self.eps)

        loss_diff = torch.mean(diff_weight.unsqueeze(-1) * (diff_pred - diff_true) ** 2)

        # Total loss
        return mse_loss + self.alpha * loss_diff

class HybridGradientNeighborMSELoss(BaseLossFunction):
    r"""
    Hybrid Gradient and Neighbor Mean Squared Error loss function.
    The loss combines the gradient-weighted MSE with the neighbor difference term.
    This encourages the model to fit the target values at each point while also considering the importance of gradients and the consistency of differences between neighboring points.

    Args:
        alpha: Weighting factor for the gradient term (default: ``1.0``).
        beta: Weighting factor for the neighbor difference term (default: ``1.0``).
        geom_dim: Number of geometric input dimensions to consider for the gradient (e.g., 3 for 3D spatial coordinates; default: ``3``).
        eps: Small constant to prevent division by zero when normalizing the neighbor differences (default: ``1e-8``).
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        geom_dim: int = 3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.geom_dim = geom_dim
        self.eps = eps

    def forward(
        self,
        model: torch.nn.Module,
        batch
    ) -> torch.Tensor:

        center_x, neighbor_x, center_y, neighbor_y = batch["x"], batch["x_neighbors"], batch["y"], batch["y_neighbors"]
        B, K, input_dim = neighbor_x.shape

        # Predict center and neighbor outputs with gradient tracking for center points
        center_x_req = center_x.clone().detach().requires_grad_(True)
        pred_center = model(center_x_req)
        pred_neighbors = model(neighbor_x.view(-1, input_dim))
        pred_neighbors = pred_neighbors.view(B, K, -1) 

        # Gradient-weighted MSE
        mse_pointwise = (pred_center - center_y) ** 2

        grads_full = torch.autograd.grad(
            outputs=pred_center,
            inputs=center_x_req,
            grad_outputs=torch.ones_like(pred_center),
            create_graph=True,
            retain_graph=True,
        )[0]

        grads_geom = grads_full[:, :self.geom_dim]
        grad_norm = torch.norm(grads_geom, dim=1)
        grad_weight = grad_norm / (grad_norm.mean() + self.eps)

        loss_weighted_mse = torch.mean((1.0 + self.alpha * grad_weight.unsqueeze(-1)) * mse_pointwise)

        # Neighbor difference term
        diff_true = neighbor_y - center_y.unsqueeze(1)
        diff_pred = pred_neighbors - pred_center.unsqueeze(1)

        diff_weight_neighbors = torch.norm(diff_true, dim=-1)
        diff_weight_neighbors = diff_weight_neighbors / (diff_weight_neighbors.mean() + self.eps)

        loss_neighbor = torch.mean(diff_weight_neighbors.unsqueeze(-1) * (diff_pred - diff_true) ** 2)

        # Total loss
        return loss_weighted_mse + self.beta * loss_neighbor