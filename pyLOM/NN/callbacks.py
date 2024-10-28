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
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
       
    def early_stop(self, validation_loss, prev_train, train):
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