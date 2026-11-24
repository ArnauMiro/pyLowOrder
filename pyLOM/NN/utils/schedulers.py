#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines.
#
# Last rev: 24/11/2026

class betaLinearScheduler:
    r"""
    Linear scheduler for beta parameter in the loss function of the Autoencoders.

    Args:
        start_value (float): value of beta until ``start_epoch``
        end_value (float): final value of beta
        start_epoch (int): epoch where beta starts growing linearly
        warmup (int): epoch where beta reaches ``end_value``
    """

    def __init__(self, start_value, end_value, start_epoch, warmup):
        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.warmup = warmup

    def getBeta(self, epoch):
        r"""
        Get the value of beta for a given epoch.

        Args:
            epoch (int): current epoch
        """
        if epoch < self.start_epoch:
            return self.start_value
        else:
            if epoch < self.warmup:
                beta = self.start_value + (self.end_value - self.start_value) * (
                    epoch - self.start_epoch
                ) / (self.warmup - self.start_epoch)
                return beta
            else:
                return self.end_value