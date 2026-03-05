#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN loss functions.
#
# Last rev: 05/03/2026

import torch 

from abc            import ABC, abstractmethod

class BaseLossFunction(torch.nn.Module, ABC):
    """
    Base abstract class for loss functions. 
    All loss functions must inherit from this class and implement the forward method, which computes the loss given a model and a batch of data.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, model, batch):
        pass

class TorchLossAdapter(BaseLossFunction):
    """
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