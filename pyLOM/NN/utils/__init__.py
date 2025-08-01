#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines.
#
# Last rev:

# Built-in modules
import os
import random
from typing import Union, Dict, Tuple, List

# Third-party libraries
import gc
import numpy as np
import torch
from torch import Tensor

# Local modules
from .. import DEVICE
from .optuna_utils import set_seed, create_results_folder, select_device, count_trainable_params, cleanup_tensors, hyperparams_serializer, get_optimizing_value
from .scalers import MinMaxScaler
from .schedulers import betaLinearScheduler
from .configs import GNSConfig, GNSTrainingConfig

from .stats         import RegressionEvaluator
from .callbacks     import EarlyStopper