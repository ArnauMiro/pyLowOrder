#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN schedulers.
#
# Last rev: 23/11/2026

class betaLinearScheduler:
    r"""
    Linear scheduler for the beta parameter in the loss function of the Autoencoders: beta is 0 before
    ``start_epoch``, grows linearly from ``start_value`` at ``start_epoch`` to ``end_value`` at ``warmup``,
    and stays at ``end_value`` afterwards.

    Args:
        start_value (float): Value of beta at ``start_epoch``.
        end_value (float): Final value of beta.
        start_epoch (int): Epoch at which beta starts to grow (0 before it).
        warmup (int): Epoch at which beta reaches ``end_value`` (an epoch, not a number of epochs).
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
            epoch (int): Current epoch.

        Returns:
            float: Value of beta.
        """
        if epoch < self.start_epoch:
            return 0
        else:
            if epoch < self.warmup:
                beta = self.start_value + (self.end_value - self.start_value) * (
                    epoch - self.start_epoch
                ) / (self.warmup - self.start_epoch)
                return beta
            else:
                return self.end_value