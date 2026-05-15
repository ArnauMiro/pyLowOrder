#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN callback routines.
#
# Last rev: 02/10/2024
from __future__ import print_function

import numpy as np


## Early stopper callback
class EarlyStopper:
    r"""
    Early stopper callback.

    Args:
        patience (int): Number of epochs to wait before stopping the training. Default: ``1``.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: ``0``.
    """
    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
       
    def early_stop(self, validation_loss: float, prev_train: float, train: float) -> bool:
        r"""
        Early stopper routine. The training will stop if the validation loss does not improve for a number of epochs.

        Args:
            validation_loss (float): Validation loss.
            prev_train (float): Previous training loss.
            train (float): Current training loss.

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif prev_train < train:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False