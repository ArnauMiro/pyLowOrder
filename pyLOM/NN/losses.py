#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Custom loss functions for NN Module
#
# Last rev: 02/10/2025

from __future__ import annotations

import torch
from torch import nn


class FocalMSELoss(nn.Module):
    """
    Focal-style Mean Squared Error for regression.

    Emphasizes harder examples by scaling the squared error with |error|^gamma.
    Useful on imbalanced targets (e.g., Cp heavily concentrated near 0) to prevent
    trivial solutions dominating the gradient.

    Args:
        gamma (float): Focusing parameter (>=0). Common values: 1.0–2.0.
        reduction (str): 'mean' (default) or 'sum'.
        eps (float): Small epsilon to avoid zero weights.
    """

    def __init__(self, gamma: float = 1.0, reduction: str = 'mean', eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        mse = diff.pow(2)
        weights = (diff.abs() + self.eps).pow(self.gamma)
        loss = weights * mse
        if self.reduction == 'sum':
            return loss.sum()
        return loss.mean()

