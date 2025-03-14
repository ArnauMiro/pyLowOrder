#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
#
# Last rev: 07/02/2025
from __future__ import print_function, division

import numpy as np


def set_random_elements_to_zero(vector, percentage):
    """
    Randomly sets a specified percentage of elements in a vector to zero.

    Args:
        vector (np.ndarray): Input vector.
        percentage (float): Percentage of elements to set to zero.

    Returns:
        np.ndarray: Modified vector with zeros.
    """
    modified_vector = vector.copy()
    num_zeros = int(len(vector) * percentage / 100)
    indices = np.random.choice(len(vector), num_zeros, replace=False)
    modified_vector[indices] = 0
    return modified_vector
