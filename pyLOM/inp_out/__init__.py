#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# IO Module
#
# Last rev: 20/10/2021

__VERSION__ = '1.0.0'

from .io_pkl  import pkl_load, pkl_save
from .io_h5   import h5_load, h5_save

del io_pkl, io_h5
